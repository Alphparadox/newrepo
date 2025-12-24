# Code Explanation (Beginner Friendly)

This document explains what each part of the code does, in simple terms.

## Files Overview

- `sample.py`: The main program you run. It downloads images, loads the LLaVA model, runs tests, and saves results.
- `helper.py`: A toolbox of functions used by `sample.py` for reading data, showing images, creating prompts, and analyzing results.
- `anil.sbatch`: The SLURM script (job file) to run `sample.py` on the HPC cluster.
- `multi_image/test/*.json`: Answer keys for the visual tasks (ground truth labels).

---

## `sample.py` (Main Program)

### 1) Settings
```python
presentation_type = "multi"  # use multi-image puzzles
difficulty_type = "kiva"     # dataset variant
output_folder = "./output"
```
These lines set how the program runs and where it saves output.

### 2) Download images
```python
os.system("wget ... ")
os.system("unzip ... ")
```
Downloads the dataset zip file and unzips it into the project folder.

### 3) Load data
```python
import helper
data_dict = helper.prepare_data(presentation_type)
```
Reads all the JSON files in `multi_image/test/` and builds `data_dict`, a dictionary where each key is an image ID and the value is the answer info (e.g., which option is correct).

### 4) Show example and build prompts
```python
helper.show_concept_example(data_dict, "2DRotation", presentation_type)
system_prompt, cross_prompt, within_prompt, extra_prompt = helper.display_all_prompts(presentation_type)
```
Displays a sample puzzle and prints the text prompts that the model will see.

### 5) Load the LLaVA model
```python
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
model = LlavaForConditionalGeneration.from_pretrained(..., device_map="auto")
```
Downloads the model and prepares it to run on the GPU(s). `device_map="auto"` spreads the model across available GPUs.

### 6) Inference helper
```python
def run_llava(prompt, image_path=None, max_tokens=150):
    inputs = processor(prompt, images=images, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.batch_decode(output, skip_special_tokens=True)[0]
```
This function sends a prompt + image(s) into the model and returns the model's text answer.

### 7) Evaluation loop
```python
for img_id, img_info in data_dict.items():
    cross_options, gt_cross, _ = helper.generate_cross_options(img_id)
    within_options, gt_within, _ = helper.generate_within_options(img_id, img_info)
    ...
    ans_cross = run_llava(final_cross_prompt, train_path)
    ans_within = run_llava(final_within_prompt, train_path)
    ans_extra = run_llava(extrapolation_prompt, [train_path, test0, test1, test2])
    ...
    results_list.append({...})
```
For each puzzle:
- Build the multiple-choice options and the correct answers
- Run the model on the training image (and test images for extrapolation)
- Save the model’s answers and the ground truth to a list

### 8) Save results
```python
pd.DataFrame(results_list).to_csv(csv_path, index=False)
```
Writes all results to `output/<presentation>_<difficulty>/all_results.csv`.

### 9) Analysis & plots
```python
analysis_results = helper.init_analysis_results(...)
helper.update_analysis_results(...)
helper.compute_accuracy_by_unique_trial(...)
helper.aggregate_by_transformation_category(...)
helper.plot_results(analysis_results, evaluation_output_folder)
```
Calculates accuracy and (optionally) creates plots summarizing performance.

---

## `helper.py` (Toolbox)

Key functions you’ll see used in `sample.py`:

- `prepare_data(presentation_type)`: Reads JSON files and returns `data_dict` with answers.
- `show_concept_example(data_dict, concept, presentation_type)`: Displays one training + three test images for a chosen concept.
- `display_all_prompts(presentation_type)`: Returns the text prompts used for Cross-Domain, Within-Domain, and Extrapolation tasks.
- `generate_cross_options(img_id)`: Returns options like `(1) Rotate`, `(2) Resize`, etc., plus the correct one.
- `generate_within_options(img_id, img_info)`: Returns options like `(1) Rotate 90°`, `(2) Rotate 180°`, etc., plus the correct one.
- `plot_results(analysis_results, output_folder)`: Saves accuracy plots.

You don’t have to edit these to run the project.

---

## `anil.sbatch` (SLURM Job)

Important lines:
```bash
#SBATCH --partition=gpu        # which nodes to use
#SBATCH --gres=shard:H100:1    # request 1 H100 GPU (your cluster may use --gres=gpu:H100:1)
#SBATCH --cpus-per-task=4      # CPU cores
#SBATCH --mem=64G              # RAM
```
This script:
1. Activates the conda environment (`kiva_env`)
2. Checks GPU status (`nvidia-smi`)
3. Runs `python sample.py`

If paths differ on your cluster, edit the last line to:
```bash
python sample.py
```

---

## JSON Files (Answer Keys)

Files like `2DRotation.json` have entries like:
```json
{
  "2DRotation+90_0_0": {
    "transform": "2DRotation+90",
    "correct": "(B)",
    "incorrect": "(C)",
    "nochange": "(A)"
  }
}
```
This means:
- The correct test option is `(B)`
- `(A)` is the "no change" option
- `(C)` is the incorrect transformation

---

## Tips

- If the model is too large for your GPU, use 8-bit loading (quantization) as shown in the README Troubleshooting section.
- Always check `logs/` for errors.
- Results live in `output/.../all_results.csv`.

---

