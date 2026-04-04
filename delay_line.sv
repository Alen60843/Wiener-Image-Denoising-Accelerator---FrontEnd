module delay_line #(
    parameter int DEPTH = 128,   // Number of elements to delay (image width)
    parameter int DW    = 8      // Data width (bits per pixel)
)
(
    input  logic          clk,
    input  logic          rst_n,
    input  logic          in_valid,      // Valid signal for input data
    input  logic [DW-1:0] in_data,       // Input data (pixel)
    output logic          out_valid,     // Valid signal for output data
    output logic [DW-1:0] out_data       // Output data (pixel delayed by DEPTH cycles)
);
    // Internal shift registers for data and valid flags
    logic [DW-1:0] shift_reg [0:DEPTH-1];
    logic          valid_reg [0:DEPTH-1];
    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize all registers on reset
            for (i = 0; i < DEPTH; i++) begin
                shift_reg[i] <= '0;
                valid_reg[i] <= 1'b0;
            end
        end else begin
            // Shift data and valid flags through the pipeline each clock
            // shift_reg[0] captures new input, shift_reg[DEPTH-1] drops off as output
            shift_reg[0] <= in_data;
            valid_reg[0] <= in_valid;
            for (i = 1; i < DEPTH; i++) begin
                shift_reg[i] <= shift_reg[i-1];
                valid_reg[i] <= valid_reg[i-1];
            end
        end
    end

    // Output the last stage of the shift register
    assign out_data  = shift_reg[DEPTH-1];
    assign out_valid = valid_reg[DEPTH-1];
endmodule
