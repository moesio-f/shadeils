This source code implement the winner of the Large-Scale Global Optimization Competition organized in IEEE Congress of Evolutionary Computation 2018, http://www.tflsgo.org/special_sessions/cec2018.html

The implementation is done in Python 3, using numpy.

This source code is freely available under the General Public License (GPLv3). However, if you use it in a research paper, you should refer to the original work:

"Molina, D., LaTorre, A. Herrera, F. SHADE with Iterative Local Search for Large-Scale Global Optimization. Proceeding of the 2018, IEEE Congress on Evolutionary Computation, Rio de Janeiro, Brasil, 8-13 July, 2018, pp 1252-1259"

It was presented in the WCCI 2018, in particular in the IEEE Congress on Evolutionary Computation. The [slides are available](https://speakerdeck.com/dmolina/shade-with-iterative-local-search-for-large-scale-global-optimization).

---
This fork fixes a few upstream bugs and update dependencies to newer versions. Additionally, it enables usage of generic fitness functions and remove dependency on the test suite of CEC2013. The code is now intended to be used as a package/library which can be further extended for specific needs while retaining most of the original implementation.

## Changelog

- (01/06/2022) Remove unused scripts for installation;
- (01/06/2022) Update `requirements.txt` to use newer versions;
- (01/06/2022) Refactor `ea` directory;
- (03/06/2022) Create new package: `shade_ils`;
- (03/06/2022) Add support for generic fitness functions;
- (03/06/2022) Remove direct CLI support from `shadeils.py`;
- (03/06/2022) Add new parameters (`evals_gs`, `evals_de`, `evals_ls`) to `ihshadels(...)`;
- (04/06/2022) Improve logging (add keys, standardize, formatting, etc);