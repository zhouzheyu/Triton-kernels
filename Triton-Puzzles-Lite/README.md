# Triton Puzzles Lite

Modified from [Triton-Puzzles](https://github.com/srush/Triton-Puzzles/) by Sasha Rush and others, which is a good educational notebook for learning Triton compiler. Triton Puzzles Lite is a lite version of Triton Puzzles, decoupling it from many unnecessary dependencies and making it more accessible for beginner users.

## Get Started


#### Installation

You only need to install `torch`. Triton and NumPy are installed when installing PyTorch. Other dependencies are fully removed from the original version. All puzzles are executed on **CPU through Triton interpreter**. So any GPU-related configuration is not necessary (i.e. You can just install torch-cpu) -- but you can also run them in GPU.

```bash
# In your Python virtual environment / conda environment
pip install torch==2.5.0
# Check triton version: triton==3.1.0
```

Note: there is a known version issue that you may encounter. See [Known Issues](#known-issues).

#### Do Puzzles

The main content is integrated in this single file: `puzzles.py`. Read through it in order and complete the exercises along the way, and you will have completed the challenge!

Run puzzles (Remeber to open the Triton interpreter mode):
```bash
# Run all puzzles. Stop at the first failed one
TRITON_INTERPRET=1 python puzzles.py -a
# Run on GPU
python puzzles.py -a
# Only run puzzle 1
TRITON_INTERPRET=1 python puzzles.py -p 1
# More arguments, refer to help
python puzzles.py -h
```

Run answers (The answer is placed in `puzzles_ans.py` for reference):
```bash
TRITON_INTERPRET=1 python puzzles_ans.py -a
```

![](imgs/all_tests_passed.png)

Check `puzzles.md` for the puzzle descriptions w/ pictures.

## Debug

1. You can use `print` to directly print the intermediate values / their shapes in Triton interpreter mode (CPU) to debug. For GPU, use `tl.static_print` or `tl.device_print` instead. See the [offcial documentation for debug Ops](https://triton-lang.org/main/python-api/triton.language.html#debug-ops).

```python
# In the Triton kernel program
print("Weight: ", weight) # Print "weight" tensor
>>> Weight:  [[ 64402 -53811   5705 -76124 -35175  42429  58555 -77519]
 [-33853  47714 -48076 -94579  29209 -80145  31319 -63292]
 [ 21065 -78242  81508 -47279 -71214 -45587 -80386  59789]
 [ 39031  60930 -11005  99305 -23686 -51177 -99270 -94698]
 [-58890  69804 -65105 -32702 -46150  27603  48390 -54706]
 [ 39536 -39587 -38564  27663 -20774 -16824   8992  46506]
 [-24661 -21011  89191  49598  -8730 -95667 -42347 -97858]
 [ -9297  81289  59782 -75179 -30261 -11214 -67609 -46084]
 [ 67937 -74551 -45982 -47662 -51844 -23186 -20091  -6341]
 [ -1770  25156 -37889 -98371  50066  -4516 -95346 -24835]
 [-60398 -68031  91756 -12160 -57719  -4944  99426 -85976]
 [ -2413 -55587 -55574 -42096  30394 -25157  66776  83608]
 [ 96417 -18400 -23771  38072 -57150  97775 -60829  59804]
 [-75186 -44539 -87349  71411  69624 -45786 -71564 -48474]
 [  5820 -19245 -45722  64354 -72452 -60228 -36410 -92923]
 [ 75931 -99995 -21683 -62615 -21116  32662 -52115 -97739]]

print("Weight Shape: ", weight.shape) # Print the shape of "weight" tensor
>>> Weight Shape:  [constexpr[16], constexpr[8]]
```

2. For better debugging, we enhance the test function to print more information. If your output is different from the expected output, we will print them as well as the positions of the different values:

![](imgs/diff_output.png)

3. And if invalid memory access is detected, we will print the memory access information, including access offsets and valid/invalid mask. (This is implemented by hooking the Triton interpreter, so only in CPU mode):

![](imgs/invalid_mem_access.png)

## Changes

- For minimal dependency, we remove the visualization part of Triton Puzzles and turn it from a Jupyter notebook to a Python script.

- Fix some problem descriptions. 
    - Puzzle 6: `(i, j)` should be `(j, i)`, and `x` should be two-dimensional.
    - Puzzle 9: The original description and notations are confusing. Change to the correct version.
    - Puzzle 10: Change the index var `k` to `l` to avoid confusing with the kernel `k`.
    - Puzzle 12: Add some notes about the difference of `shift` in the formula and the real tests.

- Small modifications to the test function & triton-viz interpreter, for better debugging.

- Some minor modifications in the puzzles code (Mainly for better readability, e.g. variable naming).

## Known Issues
<a id="known_issues"></a>

- Puzzle 11, 12 fail in GPU mode.

- There are some campatibility issues with Triton interpreter and NumPy 2.0. To check it, you can first run demos to see whether the results are correct:
    ```bash
    TRITON_INTERPRET=1 python3 puzzles.py -i
    ```
    If the results of Demo 1 look like:
    ```
    Demo1 Output: 
    [0 1 2 3 4 5 6 7]
    [0. 0. 0. 0. 0. 0. 0. 0.]
    ```
    Then you should first fix your version problem. You can read [this issue](https://github.com/SiriusNEO/Triton-Puzzles-Lite/issues/1) for a detailed solutions.
