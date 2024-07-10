import os
import re
import argparse

def combine_output_files(folder:str, output_file:str):
    """
    Combines all the output files in the folder which fulfill the regex  into a single file
    """
    #precompile the regex
    file_regex = re.compile("evaluation_results_avg[-_](.*)[-_](.*).csv")
    is_first = True
    with open(output_file, "w") as out_file:
        for file in os.listdir(folder):
            # extract the regex groups if the file matches the regex
            match = file_regex.match(file)
            if match:
                type_model, type_dataset = match.groups()
                with open(os.path.join(folder, file)) as in_file:
                    first_line = in_file.readline()
                    if is_first:
                        out_file.write("dataset,method," + first_line)
                        is_first = False
                    for line in in_file:
                        out_file.write(type_dataset + "," + type_model + "," + line)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", help="Folder containing the output files", default="./output")
    parser.add_argument("--output_file", "-o", help="Output file", default="./output/combined_avg_output.csv")
    args = parser.parse_args()
    
    combine_output_files(args.folder, args.output_file)