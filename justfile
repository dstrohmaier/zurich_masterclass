compile_presentations:
    jupyter nbconvert presentations/*.ipynb --to slides # -post serve


## embedding operation

store_w2v:
    python run_operations.py reduce_w2v