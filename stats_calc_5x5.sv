module stats_calc_5x5 #(
    parameter int DW = 8   // Pixel data width (bits)
)(
    input  logic          clk,
    input  logic          rst_n,
    input  logic          stats_in_valid,   // Asserted when all 25 pixel inputs are valid for computation
    input  logic          mode_3x3,    // If high, compute stats for center 3x3 pixels only; else compute for full 5x5
    input  logic [31:0]   divider,    // Divider value: 9 for 3x3 mode, 25 for 5x5 mode

    // 25 pixel inputs from the 5x5 window, named by row_offset (m2, m1, 0, p1, p2) and col_offset
    input  logic [DW-1:0] p_m2_m2, p_m2_m1, p_m2_0,  p_m2_p1, p_m2_p2,
    input  logic [DW-1:0] p_m1_m2, p_m1_m1, p_m1_0,  p_m1_p1, p_m1_p2,
    input  logic [DW-1:0] p_0_m2,  p_0_m1,  p_0_0,   p_0_p1,  p_0_p2,
    input  logic [DW-1:0] p_p1_m2, p_p1_m1, p_p1_0,  p_p1_p1, p_p1_p2,
    input  logic [DW-1:0] p_p2_m2, p_p2_m1, p_p2_0,  p_p2_p1, p_p2_p2,

    output logic          out_valid,  // Goes high when output mean and variance are valid (1 cycle after in_valid)
    output logic [15:0]   mu_q8_8,    // Mean of 25 pixels (16-bit Q8.8 fixed-point)
    output logic [23:0]   var_q16_8   // Variance of 25 pixels (24-bit Q16.8 fixed-point)
);
    // 32-bit accumulators to avoid overflow (25 * max_pixel_value^2 fits in 32 bits since max 255^2 * 25 < 2^31)
    logic [31:0] sum_x;
    logic [31:0] sum_x2;
    
    logic [15:0] mu_c;
    logic [23:0] ex2_c;
    logic [23:0] mu_sq_c;

    //map inputs into an array for compact looping
    logic [7:0]  pix [0:24];       // will hold the 25 pixel values
    logic [31:0] sq  [0:24];       // will hold the square of each pixel (32-bit each)

    // Delay the sum and square sum calculations by 1 cycle to handle setup time
    logic [31:0] sum_x_d;
    logic [31:0] sum_x2_d;

    // Indicator that tell when we can sample the sums 
    logic sums_computed;

    // delay the stat_in_valid by 1 cycle to align with sums
    logic stats_in_valid_d;

    always_comb begin
        // Map Inputs into pix array
        pix[0]  = p_m2_m2;  pix[1]  = p_m2_m1;  pix[2]  = p_m2_0;  pix[3]  = p_m2_p1;  pix[4]  = p_m2_p2;
        pix[5]  = p_m1_m2;  pix[6]  = p_m1_m1;  pix[7]  = p_m1_0;  pix[8]  = p_m1_p1;  pix[9]  = p_m1_p2;
        pix[10] = p_0_m2;   pix[11] = p_0_m1;   pix[12] = p_0_0;   pix[13] = p_0_p1;   pix[14] = p_0_p2;
        pix[15] = p_p1_m2;  pix[16] = p_p1_m1;  pix[17] = p_p1_0;  pix[18] = p_p1_p1;  pix[19] = p_p1_p2;
        pix[20] = p_p2_m2;  pix[21] = p_p2_m1;  pix[22] = p_p2_0;  pix[23] = p_p2_p1;  pix[24] = p_p2_p2;
        

        // Calculate squares of each pixel
        for (int i = 0; i < 25; i++) begin
            sq[i] = 32'(pix[i]) * 32'(pix[i]);  // square each pixel value (result fits in 16 bits, but use 32 bits for sum stability)
        end

        // Summation of all pixel values and all squared values
        sum_x  = 32'd0;
        sum_x2 = 32'd0;
        for (int i = 0; i < 25; i++) begin
            if (mode_3x3 && 
               ( (i == 6)  || (i == 7)  || (i == 8) ||
                 (i == 11) || (i == 12) || (i == 13) ||  
                 (i == 16) || (i == 17) || (i == 18) ) ) begin
                    sum_x  += 32'(pix[i]);  // sum of pixel intensities
                    sum_x2 += sq[i];        // sum of squared intensities
                 end
            else if (!mode_3x3) begin
                    sum_x  += 32'(pix[i]);  
                    sum_x2 += sq[i];       
                end
        end
        end

        // Delay sums by 1 cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_x_d  <= 32'd0;
            sum_x2_d <= 32'd0;
            sums_computed <= 1'b0;
            stats_in_valid_d <= 1'b0;
        end else begin
            sum_x_d  <= sum_x;
            sum_x2_d <= sum_x2;
            sums_computed <= 1'b1;
            stats_in_valid_d <= stats_in_valid;
        end
    end

        // Calculate Mean (Q8.8 fixed-point)
        // Mean = (sum_x * 256) / 25. 
        // Multiplying by 256 (<< 8) shifts the result to Q8.8 format before dividing by 25.
        assign mu_c = (sum_x_d * 32'd256) / divider;
        // Calculate E[x^2] (expected value of squared intensity, Q16.8)
        // E[x^2] = (sum_x2 * 256) / 25, yielding a Q16.8 result.
        assign ex2_c = (sum_x2_d * 32'd256) / divider;
        // Calculate (E[x])^2 (mu squared, in Q16.8)
        // mu_c is Q8.8, so convert to Q16.16 by multiplying, then >> 8 to go to Q16.8.
        assign mu_sq_c = (32'(mu_c) * 32'(mu_c)) >> 8;
    

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            mu_q8_8   <= '0;
            var_q16_8 <= '0;
        end 
        else if (sums_computed) begin
            out_valid <= stats_in_valid_d;   // Output is valid one cycle after input was valid 
            mu_q8_8   <= mu_c;       // Register the mean result
            // Variance = E[x^2] - (E[x])^2 (in same Q16.8 format)
            if (ex2_c > mu_sq_c)
                var_q16_8 <= ex2_c - mu_sq_c;
            else
                var_q16_8 <= 24'd0;  // Variance can't be negative (if rounding causes mu_sq_c to slightly exceed ex2_c, clamp to 0)
        end
    end

endmodule