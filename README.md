This source code implement the winner of the Large-Scale Global Optimization
Competition organized in IEEE Congress of Evolutionary Computation 2018, 
http://www.tflsgo.org/special_sessions/cec2018.html

The implementation is done in Python 3, using numpy.

This source code is freely available under the General Public License (GPLv3).
However, if you use it in a research paper, you should refer to the original 
work:

"Molina, D., LaTorre, A. Herrera, F. SHADE with Iterative Local Search for
Large-Scale Global Optimization. Proceeding of the 2018, IEEE Congress on Evolutionary
Computation, Rio de Janeiro, Brasil, 8-13 July, 2018, pp 1252-1259"

It was presented in the WCCI 2018, in particular in the IEEE Congress on
Evolutionary Computation. The
[slides are available](https://speakerdeck.com/dmolina/shade-with-iterative-local-search-for-large-scale-global-optimization).

---
This fork fixes a few upstream bugs during installation and update all required dependecies to newer versions.

## Changelog

- (01/06/2022) Remove unused scripts for installation;
- (01/06/2022) Update `requirements.txt` to use newer versions;
- (01/06/2022) Refactor `ea` directory;