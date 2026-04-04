module col_reflect_block #(
    parameter int DW = 8
)(
    input  logic          clk,
    input  logic          rst_n,
    input  logic          sel_reflect,         // Select reflected column when 1, normal column when 0
    input  logic          en,

    // Each column is a packed array of 5 pixels (one for each row of the 5x5 window)
    input  logic [4:0][DW-1:0] normal,          // Normal column pixels (in order from top row to bottom row)
    input  logic [4:0][DW-1:0] reflect,         // Reflected column pixels

    output logic [4:0][DW-1:0] out_col          // Output column (5 pixels)
);
    logic [4:0][DW-1:0] out_col_c;
    genvar i;

    generate
        // For each of the 5 pixel positions in the column, choose reflect or normal based on sel_reflect
        for (i = 0; i < 5; i++) begin : COL
            assign out_col_c[i] = sel_reflect ? reflect[i] : normal[i];
        end
    endgenerate

    // Register the output column
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_col <= '0;
        end else if (en) begin
            out_col <= out_col_c;
        end
    end
endmodule
