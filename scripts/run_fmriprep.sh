singularity run --cleanenv --bind /home:/data \
    /home/nastase/social_actions/singularity/fmriprep-v20.1.1.simg \
    --participant-label sub-$1 \
    --nthreads 8 --omp-nthreads 8 \
    --longitudinal --bold2t1w-dof 6 \
    --medial-surface-nan \
    --output-spaces fsaverage6 \
    --notrack \
    --use-syn-sdc --write-graph --work-dir \
    /data/nastase/social_actions/fmri/1021_actions/derivatives/work \
    /data/nastase/social_actions/fmri/1021_actions \
    /data/nastase/social_actions/fmri/1021_actions/derivatives participant 
