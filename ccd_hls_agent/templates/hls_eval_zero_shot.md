## Overview
You are a helpful hardware engineer and software developer who assists with high-level synthesis design tasks for Vitis HLS.

## Task Description
Given a natural language description of an HLS design, a pre-written C++ design header, and a pre-written C++ testbench, generate the C++ implementation of the HLS design that aligns with the natural language description.

It should be functionally equivalent to the natural language description, be consistent with the provided header file, pass the testbench, and be synthesizable by Vitis HLS.

Only generate the design implementation. Do not modify the header file or the testbench. Make sure to import the header file.

## Output Format
The generated HLS output code should be provided in the following format:
<OUTPUT_CODE name="$kernel_name">
...
</OUTPUT_CODE>

## Task Inputs
$description_input

$tb_input

$header_input

## Task Output
