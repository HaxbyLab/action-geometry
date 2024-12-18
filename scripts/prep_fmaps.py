from os.path import join
import json
from glob import glob

bids_dir = '/home/nastase/social_actions/fmri/1021_actions'

subjects = ['sub-sid000005', 'sub-sid000007', 'sub-sid000009',
            'sub-sid000010', 'sub-sid000012', 'sub-sid000013',
            'sub-sid000020', 'sub-sid000021', 'sub-sid000024',
            'sub-sid000029', 'sub-sid000034', 'sub-sid000052',
            'sub-sid000102', 'sub-sid000114', 'sub-sid000120',
            'sub-sid000134', 'sub-sid000142', 'sub-sid000278',
            'sub-sid000416', 'sub-sid000433', 'sub-sid000499',
            'sub-sid000522', 'sub-sid000535']

sessions = ['ses-actions1', 'ses-actions2', 'ses-raiders']

# Loop through subjects and sessions
for subject in subjects:
    for session in sessions:

        # Compile and check all four runs
        func_fns = sorted(glob(join(bids_dir, subject, session, 'func',
                                    f'{subject}_{session}_*bold.nii.gz')))
        assert len(func_fns) == 4
        for i, func_fn in enumerate(func_fns):
            assert f'{i+1:02}' == func_fn.split('run-')[-1][:2]

        func_fns = ['/'.join(fn.split('/')[-3:]) for fn in func_fns]

        fmap_jsons = glob(join(bids_dir, subject, session, 'fmap',
                               f'{subject}_{session}_*.json'))

        for fmap_json in fmap_jsons:
            with open(fmap_json) as f:
                fmap = json.load(f)

            fmap['IntendedFor'] = func_fns

            with open(fmap_json, 'w') as f:
                json.dump(fmap, f, indent=2, sort_keys=True)
