## Behaviorally-relevant features of observed actions dominate cortical representational geometry in natural vision

#### Overview 
This repo accompanies a preprint titled "Behaviorally-relevant features of observed actions dominate cortical representational geometry in natural vision" by Jane Han, Vassiki Chauhan, Rebecca Philip, Morgan Taylor, Heejung Jung, Yaroslav O. Halchenko, M. Ida Gobbini, James V. Haxby, and Samuel A. Nastase (based on a dissertation project by [Nastase, 2018](https://search.proquest.com/docview/2018905893))

Han, J., Chauhan, V., Philip, R., Taylor, M. K., Jung, H., Halchenko, Y. O., Gobbini, M. I., Haxby, J. V.\* & Nastase, S. A.\* (2024). Behaviorally-relevant features of observed actions dominate cortical representational geometry in natural vision. *bioRxiv*. https://doi.org/10.1101/2024.11.26.624178

We collected fMRI data using a condition-rich design in which participants viewed 90 naturalistic video clips depicting humans performing a variety of social and nonsocial behaviors broadly sampling action space (e.g., conversation, cooking, gardening; [Haxby et al., 2020](https://doi.org/10.1016/j.neuroimage.2020.116561)). Our analysis pipeline relies on representational similarity analysis ([Kriegeskorte et al., 2008](https://doi.org/10.3389/neuro.06.004.2008)) and surface-based whole-cortex hyperalignment based on an independent movie dataset ([Guntupalli et al., 2016](https://doi.org/10.1093/cercor/bhw068)).

![Alt text](./figure_github.png?raw=true&s=100 "Variance partitioning")

#### Software requirements
The code used for the manuscript requires two separate environments: `action-python2.yml` Python 2 environment for PyMVPA, and `action-python3.yml` Python 3 environment. In each script, we specify the corresponding environment.

#### Demo
We provide a smaller-scale demo notebook `action_geometry_demo.ipynb` to replicate the core results testing nine representational models against neural representational geometries from nine regions of interest. The demo requires Python 3, NumPy, SciPy, Pandas, Matplotlib, and Seaborn (all installed in the `action-python3.yml` environment).

#### References

* Guntupalli, J. S., Hanke, M., Halchenko, Y. O., Connolly, A. C., Ramadge, P. J., & Haxby, J. V. (2016). A model of representational spaces in human cortex. *Cerebral Cortex*, *26*(6), 2919-2934. https://doi.org/10.1093/cercor/bhw068

* Haxby, J. V., Gobbini, M. I., & Nastase, S. A. (2020). Naturalistic stimuli reveal a dominant role for agentic action in visual representation. *NeuroImage*, 116561. https://doi.org/10.1016/j.neuroimage.2020.116561

* Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis&mdash;connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, *2*, 4. https://doi.org/10.3389/neuro.06.004.2008

* Nastase, S. A. (2018). *The geometry of observed action representation during natural vision* (Publication No. 10745463) [Doctoral dissertation, Dartmouth College]. ProQuest Dissertations Publishing. https://search.proquest.com/docview/2018905893

#### Citation
If you use this code for your research, please cite the paper: https://doi.org/10.1101/2024.11.26.624178
