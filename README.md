## Behaviorally-relevant features of observed actions dominate cortical representational geometry in natural vision

#### Overview 
This repo accompanies a preprint titled "Behaviorally-relevant features of observed actions dominate cortical representational geometry in natural vision" by Jane Han, Vassiki Chauhan, Rebecca Philip, Morgan Taylor, Heejung Jung, Yaroslav O. Halchenko, M. Ida Gobbini, James V. Haxby, and Samuel A. Nastase:

Han, J., Chauhan, V., Philip, R., Taylor, M. K., Jung, H., Halchenko, Y. O., Gobbini, M. I., Haxby, J. V.\* & Nastase, S. A.\* (2024). Behaviorally-relevant features of observed actions dominate cortical representational geometry in natural vision. *bioRxiv*. https://doi.org/10.1101/2024.11.26.624178

We collected fMRI data using a condition-rich design in which participants viewed 90 naturalistic video clips depicting humans performing a variety of social and nonsocial behaviors broadly sampling action space (e.g., conversation, cooking, gardening; [Haxby et al., 2020](https://doi.org/10.1016/j.neuroimage.2020.116561)). Our analysis pipeline relies on representational similarity analysis ([Kriegeskorte et al., 2008](https://doi.org/10.3389/neuro.06.004.2008)) to compare how well different representational models capture cortical representational geometry.

<br>

![Alt text](./figure_github.png?raw=true&s=100 "Variance partitioning")

<br>

#### Software requirements
The code used for the manuscript requires two separate environments: `action-python2.yml` Python 2 environment for PyMVPA, and `action-python3.yml` Python 3 environment. In each script, we specify the corresponding environment.

#### Demo
We provide a small-scale demo notebook [`action_geometry_demo.ipynb`](https://github.com/HaxbyLab/action-geometry/blob/add-demo/action_geometry_demo.ipynb) to replicate the core results testing nine representational models against neural representational geometries from nine regions of interest. The demo requires Python 3, Jupyter, NumPy, SciPy, Pandas, Matplotlib, and Seaborn. You can set up a dedicated conda environment for running the demo with the following code (should only take a couple minutes on a typical laptop):

```
conda create -n action-demo
conda activate action-demo
conda install -c conda-forge jupyterlab numpy scipy pandas matplotlib seaborn
```

#### References

* Han, J., Chauhan, V., Philip, R., Taylor, M. K., Jung, H., Halchenko, Y. O., Gobbini, M. I., Haxby, J. V.\* & Nastase, S. A.\* (2024). Behaviorally-relevant features of observed actions dominate cortical representational geometry in natural vision. *bioRxiv*. https://doi.org/10.1101/2024.11.26.624178

* Haxby, J. V., Gobbini, M. I., & Nastase, S. A. (2020). Naturalistic stimuli reveal a dominant role for agentic action in visual representation. *NeuroImage*, 116561. https://doi.org/10.1016/j.neuroimage.2020.116561

* Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis&mdash;connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, *2*, 4. https://doi.org/10.3389/neuro.06.004.2008

#### Citation
If you use this code for your research, please cite the paper: https://doi.org/10.1101/2024.11.26.624178
