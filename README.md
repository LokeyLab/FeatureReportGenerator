# Feature Report Generator (Rust implementation)

#### Original code created by Akshar Lohith
#### Converted to CLI and optimized by Derfel Terciano

## Description/Program Information

This program generates a feature report (in the form of an Excel Workbook) from a given experimental dataset and a reference dataset (both of which must be .csv files).

In the final output, the Excel workbook contains multiple sheets where each sheet corresponds to one of the experimental wells/compound IDs in the given experimental dataset. Each sheet contains all the reference wells/compounds IDs and each well corresponds to a calculated Pearson R score or a Correlation distance relative to the experimental well.

***Note:*** This program uses parallel processing (thanks to rayon) and multithreading in order to produce an output in a reasonable amount of time. Since converting to rust, this program is extremely fast and can **officially** handle large datasets!

## Usage/Help

#### How to compile program
1) Make sure to have rust installed. If not you can use [Rustup](https://www.rust-lang.org/tools/install) to install the rust compiler and toolchain.
2) An optional `Makefile` is included in this repo. You can use make (to build) and make clean (to clean and remove all build/binaries) to compile or remove the binaries.
    1) Once compiled, a symlink to actual binary is generated in the root of this repo.
3) **If** you don't have `make` installed on your system, you can manually compile the binary yourself.
    1) Run in the root of this repo ``cargo build --release``
    2) The executable binary is located in `target/release/FeatureReportGenerator` 
    3) You can take the binary out of the directory above and execute it anywhere however, it is best practice to make a symlink to the actual binary instead of moving the actual binary out.
4) You can now execute the binary using `./feat_report_gen.binary -h`

#### Using the program
Calling `./feat_report_gen.binary -h` produces the following:

    ➜ ./feat_report_gen.binary -h
    Generates a similarity report between a experimental dataset and a reference set

    Usage: feat_report_gen.binary [OPTIONS] --exp-input <EXP_INPUT> --ref-input <REF_INPUT> --outpath <OUTPATH> --index <INDEX>

    Options:
    -e, --exp-input <EXP_INPUT>  Path to experimental dataset (.csv format)
    -r, --ref-input <REF_INPUT>  Path to reference dataset (.csv format)
    -o, --outpath <OUTPATH>      Output file path (must end in .xlsx)
    -t, --threads <THREADS>      Number of threads to use (optional)
    -i, --index <INDEX>          index column
    -d, --distance               If distance correlation (default) or pearson r should be used
    -h, --help                   Print help
    -V, --version                Print version