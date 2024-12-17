datalad containers-run \
    --explicit \
    -n fmriprep/containers/bids-fmriprep \
    --input fmriprep/sourcedata/sub-$1 \
    --output fmriprep -- \
    --participant-label $1 \
    --nthreads 8 --omp-nthreads 8 \
    --longitudinal --bold2t1w-dof 6 \
    --medial-surface-nan \
    --output-spaces fsaverage6 \
    --notrack --use-syn-sdc \
    --fs-license-file $PWD/containers/licenses/fs-license.txt \
    --skip-bids-validation \
    --write-graph --work-dir fmriprep/work \
    fmriprep/sourcedata fmriprep/ participant 
