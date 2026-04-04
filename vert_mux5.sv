module vert_mux5 #(
    parameter int DW = 8
)(
    input  logic           clk,
    input  logic           rst_n,

    // Edge condition indicators
    input  logic           top0,      // center row = 0
    input  logic           top1,     // center row = 1
    input  logic           bottom1,  // center row = max_row -2
    input  logic           bottom0,  // center row = max_row -1

    // Inputs from line buffers / stream:
    // v0 = Stream (Row +2), v1 = LB0 (Row +1), v2 = LB1 (Row 0), v3 = LB2 (Row -1), v4 = LB3 (Row -2)
    input  logic [DW-1:0]  v0_row_p2, 
    input  logic [DW-1:0]  v1_row_p1,
    input  logic [DW-1:0]  v2_row_0,
    input  logic [DW-1:0]  v3_row_m1,
    input  logic [DW-1:0]  v4_row_m2,

    output logic [DW-1:0]  row_m2,   // Output pixel for row -2 of window
    output logic [DW-1:0]  row_m1,   // Output pixel for row -1
    output logic [DW-1:0]  row_0,    // Output pixel for row  0 (center row)
    output logic [DW-1:0]  row_p1,   // Output pixel for row +1
    output logic [DW-1:0]  row_p2    // Output pixel for row +2
);
    logic [DW-1:0] row_m2_c, row_m1_c, row_0_c, row_p1_c, row_p2_c;  // combinatorial outputs

    always_comb begin
        // Default: Normal straight-through (no edge condition)
        row_m2_c = v4_row_m2;  // Use LB3 (row -2)
        row_m1_c = v3_row_m1;  // Use LB2 (row -1)
        row_0_c  = v2_row_0;   // Use LB1 (row  0)
        row_p1_c = v1_row_p1;  // Use LB0 (row +1)
        row_p2_c = v0_row_p2;  // Use stream (row +2)

        // Top Reflection: If at top edge, we cannot read above the image (rows -2, -1 don't exist)
        if (top0) begin // missing both row -2 and row -1
            row_m2_c = v0_row_p2;  // For row -2 output, use stream (row +2)
            row_m1_c = v1_row_p1;  // For row -1 output, use LB0 (row +1)
            // (Center and below remain unchanged)
        end
        else if (top1) begin // missing only row -2
            row_m2_c = v0_row_p2;  // For row -2 output, use stream (row +2)
            // (Center and below remain unchanged)
        end
        // Reflect Bottom: If at bottom0 edge, cannot read below the image (rows +1, +2 beyond image)
        else if (bottom0) begin
            row_p2_c = v4_row_m2;  // For row +2 output, use LB3 (row -2)
            row_p1_c = v3_row_m1;  // For row +1 output, use LB2 (row -1)
            // (Center and above remain unchanged)
        end
        else if (bottom1) begin
            row_p2_c = v4_row_m2;  // For row +2 output, use LB3 (row -2)
            // (Center and above remain unchanged)
        end
    end

    // Register outputs (ensures timing alignment and glitch-free outputs)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_m2 <= '0;
            row_m1 <= '0;
            row_0  <= '0;
            row_p1 <= '0;
            row_p2 <= '0;
        end else begin
            row_m2 <= row_m2_c;
            row_m1 <= row_m1_c;
            row_0  <= row_0_c;
            row_p1 <= row_p1_c;
            row_p2 <= row_p2_c;
        end
    end
endmodule
