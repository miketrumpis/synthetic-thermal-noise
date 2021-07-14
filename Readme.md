# ``synthnoise`` synthesizes electrode thermal noise

## Requires

* Python3
* numpy
* scipy

## EIS model

Thermal noise is synthesized from a model of the electrode impedance spectrum (EIS).
The electrode model is a basic two-compartment parallel resistor || constant phase element with real resistance in series. Some sample EIS measurements from a 61-channel micro-electrocorticography array are included as the default model ([Woods, Trumpis, et al., 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6342453/)  DOI 10.1088/1741-2552/aae39d).

Other models should be easy to plug-in with a callable function of frequencies.

## Timeseries synthesis

Uses the Kolmogorov spectral factorization method (based on code [here](http://web.cvxr.com/cvx/examples/filter_design/html/spectral_fact.html)).

## Demo

See the notebook! (requires installing Jupyter components)