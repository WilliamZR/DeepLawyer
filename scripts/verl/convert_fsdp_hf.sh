ckpt_path=$1

## get all ckpt folders start with "global_step", print every folder name
for folder in $(ls -d ${ckpt_path}/global_step*); do
    python verl/fsdp_to_hf.py --local_dir $folder/actor
done