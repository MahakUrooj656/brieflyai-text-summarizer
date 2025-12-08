from summarizer import TextSummarizer, compute_length_reduction
import sys
import os #helps us check if an argument is a valid file path.

# Default values
model_name = "facebook/bart-large-cnn"
max_length = 60
min_length = 20
output_file = None

# Check if user passed a --model flag
if "--model" in sys.argv:
    try:
        model_index = sys.argv.index("--model")
        selected_model = sys.argv[model_index + 1].lower()

        if selected_model == "bart":
            model_name = "facebook/bart-large-cnn"
        elif selected_model == "distilbart":
            model_name = "sshleifer/distilbart-cnn-12-6"
        elif selected_model == "t5":
            model_name = "t5-small"
        else:
            print(f"Unknown model: {selected_model}")
            sys.exit(1)

        # Remove the model option from arguments so input_text stays clean
        del sys.argv[model_index:model_index+2]

    except (IndexError, ValueError):
        print("Usage: --model <bart|distilbart|t5>")
        sys.exit(1)

# Check for --max flag (max_length)
if "--max" in sys.argv:
    try:
        max_index = sys.argv.index("--max")
        max_length = int(sys.argv[max_index + 1])  # convert string to int
        # Remove the flag and its value from sys.argv
        
        del sys.argv[max_index:max_index + 2]

    except (IndexError, ValueError):
        print("Usage: --max <integer>")
        sys.exit(1)

# Check for --min flag (min_length)
if "--min" in sys.argv:
    try:
        min_index = sys.argv.index("--min")
        min_length = int(sys.argv[min_index + 1])  # convert string to 
        
        # Remove the flag and its value from sys.argv
        del sys.argv[min_index:min_index + 2]

    except (IndexError, ValueError):
        print("Usage: --min <integer>")
        sys.exit(1)

# Check for --out flag (write summary to a file)
if "--out" in sys.argv:
    try:
        out_index = sys.argv.index("--out")
        output_file = sys.argv[out_index + 1]

        # Remove the flag and its value from sys.argv
        del sys.argv[out_index : out_index + 2]

    except (IndexError, ValueError):
        print("Output file name missing\nUsage: --out <output_filename>")
        sys.exit(1)


# Check if the user provided text as an argument
if len(sys.argv) < 2:
    print("Usage: ")
    print("python summarize_cli.py \"your long text here\"")
    print ("python summarize_cli.py path/to/file.txt")
    sys.exit(1)

arg_text= "".join(sys.argv[1:]) #combine everything after the script name into one string

input_text = None

# If there is exactly one argument and it looks like a .txt file, try to read it
if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".txt") and os.path.isfile(sys.argv[1]):
    file_path = sys.argv[1]
    print(f"Reading text from file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        input_text = f.read() #Read the entire contents of the file as a single string.
else:
    # Otherwise, treat everything as direct text input
    input_text = arg_text
     

# 2) Create a TextSummarizer object
ts = TextSummarizer(
    model_name=model_name,
    max_length=max_length,
    min_length=min_length,
    do_sample=False
)

# 3) Summarize the text
summary = ts.summarize(input_text)
orig_len, sum_len, reduction = compute_length_reduction(input_text, summary)

# 4) Print summary
print("=== STATS ===")
print(f"Original length: {orig_len} words")
print(f"Summary length:  {sum_len} words")
print(f"Reduction:       {reduction * 100:.1f}%")


# If the user provided an output file, write summary into it
if output_file:
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\nSummary saved to: {output_file}")

    except Exception as e:
        print(f"Error writing to file: {e}")

print("\n=== SUMMARY ===")
print(summary)


