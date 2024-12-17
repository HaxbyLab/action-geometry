# delimits comments

# Creation information:
#     user    : nastase
#     date    : Sun Oct 29 00:01:58 EDT 2017
#     machine : head1
#     pwd     : /home/nastase/social_actions/fmri/1021_actions/derivatives/freesurfer/fsaverage6/SUMA
#     command : @SUMA_Make_Spec_FS -NIFTI -no_ld -inflate 10 -inflate 20 -inflate 30 -inflate 40 -inflate 50 -inflate 60 -inflate 70 -inflate 80 -inflate 90 -inflate 100 -inflate 120 -inflate 150 -inflate 200 -sid fsaverage6

# define the group
        Group = fsaverage6

# define various States
        StateDef = smoothwm
        StateDef = pial
        StateDef = inflated
        StateDef = occip.patch.3d
        StateDef = occip.patch.flat
        StateDef = occip.flat.patch.3d
        StateDef = fusiform.patch.flat
        StateDef = full.patch.3d
        StateDef = full.patch.flat
        StateDef = full.flat.patch.3d
        StateDef = full.flat
        StateDef = flat.patch
        StateDef = sphere
        StateDef = white
        StateDef = sphere.reg
        StateDef = rh.sphere.reg
        StateDef = lh.sphere.reg
        StateDef = pial-outer-smoothed
        StateDef = inf_10
        StateDef = inf_20
        StateDef = inf_30
        StateDef = inf_40
        StateDef = inf_50
        StateDef = inf_60
        StateDef = inf_70
        StateDef = inf_80
        StateDef = inf_90
        StateDef = inf_100
        StateDef = inf_120
        StateDef = inf_150
        StateDef = inf_200

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.smoothwm.gii
        LocalDomainParent = SAME
        SurfaceState = smoothwm
        EmbedDimension = 3
        Anatomical = Y
        LabelDset = rh.aparc.a2009s.annot.niml.dset

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.pial.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = pial
        EmbedDimension = 3
        Anatomical = Y

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inflated.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inflated
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.sphere.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = sphere
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.white.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = white
        EmbedDimension = 3
        Anatomical = Y

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.sphere.reg.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = sphere.reg
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_10.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_10
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_20.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_20
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_30.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_30
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_40.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_40
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_50.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_50
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_60.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_60
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_70.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_70
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_80.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_80
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_90.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_90
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_100.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_100
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_120.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_120
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_150.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_150
        EmbedDimension = 3
        Anatomical = N

NewSurface
        SurfaceFormat = ASCII
        SurfaceType = GIFTI
        SurfaceName = rh.inf_200.gii
        LocalDomainParent = rh.smoothwm.gii
        SurfaceState = inf_200
        EmbedDimension = 3
        Anatomical = N
