# FeatureReportGenerator (genFeatReport.py)

#### Original code created by Akshar Lohith
#### Converted to CLI and optimized by Derfel Terciano

## Description/Program Information

This program generates a feature report (in the form of an Excel Workbook) from a given experimental dataset and a reference dataset (both of which must be .csv files).

In the final output, the Excel workbook contains multiple sheets where each sheet corresponds to one of the experimental wells/compound IDs in the given experimental dataset. Each sheet contains all the reference wells/compounds IDs and each well corresponds to a calculated Pearson R score and a Correlation distance relative to the experimental well.

***Note:*** This program uses parallel processing and multithreading in order to produce an output in a reasonable amount of time. But overall, the amount of time it takes for the program to generate the feature report is still proportional to the size of the inputted dataset/

## Usage/Help

Calling `python3 geneFeatReport.ph -h` produces usage instructions for the program:

    usage: python3 genFeatReport.py -e <experimental wells> -r <reference wells> -o <output name> [-options] [argument]

    Generates a Feature Report (xlsx file) that shows the reference compounds and
    its Correlation Distance and its Pearson R score

    options:
    -h, --help            show this help message and exit
    -e [EXPERIMENTAL], --experimental [EXPERIMENTAL]
                            file input for experimental wells (.csv file accepted
                            only)
    -r [REFERENCE], --reference [REFERENCE]
                            file input for experimental wells (.csv files accepted
                            only)
    -o [OUT], --out [OUT]
                            name for output file in xlsx format (make sure it ends
                            in .xlsx)
    -t THREADS, --threads THREADS
                            Number of threads to use for feature report writing
                            (default: 8)
    -i INDEX [INDEX ...], --index INDEX [INDEX ...]
                            Specifies which columns of the input files are the
                            index columns (default: 0 1 2 3)
    -v, --verbose         Enables verbose output to stderr (default: False)
                            Note: I recommend having this flag enabled when
                            testing or building with this program

***Note:*** make sure all libraries in `requirements.txt` are installed in system