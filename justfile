compile_presentations:
    jupyter nbconvert presentations/*.ipynb --to slides # -post serve


## embedding operation

store_w2v:
    python run_operations.py reduce_w2v



## servers

push2flamingo:
    rsync -rtvu --progress --exclude-from=../ignorelist.txt ./ ds858@flamingo.cl.cam.ac.uk:/local/scratch/ds858/zurich_masterclass

pull_flamingo:
    rsync -rtvu --progress --exclude-from=../ignorelist.txt ds858@flamingo.cl.cam.ac.uk:/local/scratch/ds858/zurich_masterclass/data ./
