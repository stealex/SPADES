Physics Background
==================

This section follows the notation and formulas in F.-A. Ghinescu and S. Stoica (2026), `arXiv:2601.10247 <https://arxiv.org/abs/2601.10247>`_.

Half-Life Decomposition
-----------------------

For two-neutrino double-beta decay, the inverse half-life is written as (paper Eq. 4):

.. math::

   \left(T_{1/2}^{2\nu}\right)^{-1} = g_A^4 \left|M^{2\nu}\right|^2 G^{2\nu}.

The decay probability factor (paper Eq. 17) is:

.. math::

   dW_{2\nu} = a_{2\nu}\,F_0(Z_f,\epsilon_1)F_0(Z_f,\epsilon_2)\,d\Omega_{2\nu}.

The differential angular-energy distribution (paper Eq. 21) is:

.. math::

   \frac{dW_{2\nu}}{d\epsilon_1\,d\cos\theta}
   = \frac{a_{2\nu}}{2\,(m_ec^2)}
   F_0(Z_f,\epsilon_1)F_0(Z_f,\epsilon_2)\,\omega_{2\nu}(\epsilon_1)\,[1+\alpha(\epsilon_1)\cos\theta].

Normalized Spectra and PSFs
---------------------------

Using the paper definitions (Eqs. 22-23), the normalized single-electron spectra are:

.. math::

   \frac{1}{G^{2\nu}}\frac{dG^{2\nu}}{d\epsilon_1}
   = \frac{\omega_{2\nu}(\epsilon_1)}{\int_1^{T+1}\omega_{2\nu}(\epsilon_1)\,d\epsilon_1},

.. math::

   \frac{1}{H^{2\nu}}\frac{dH^{2\nu}}{d\epsilon_1}
   = \frac{\chi_{2\nu}(\epsilon_1)}{\int_1^{T+1}\chi_{2\nu}(\epsilon_1)\,d\epsilon_1}.

The angle-integrated coefficient is (paper Eq. 25):

.. math::

   K^{2\nu} = \frac{H^{2\nu}}{G^{2\nu}}.

The summed-electron spectrum is (paper Eq. 26):

.. math::

   \frac{1}{G^{2\nu}}\frac{dG^{2\nu}}{dt}
   = \frac{\Sigma_{2\nu}(t)}{\int_0^T \Sigma_{2\nu}(t)\,dt}.

Transition-Dependent PSFs
-------------------------

For ``0^+ \rightarrow 0^+`` transitions, the paper gives (Eq. 35):

.. math::

   G^{2\nu}_{0^+\to0^+}
   = \frac{\mathcal{A}_2}{\ln 2}\int_1^{T+1}\omega_{2\nu}(\epsilon_1)\,d\epsilon_1.

For ``0^+ \rightarrow 2^+`` transitions (Eq. 36):

.. math::

   G^{2\nu}_{0^+\to2^+}
   = \frac{\mathcal{A}_6}{\ln 2}\int_1^{T+1}\omega_{2\nu}(\epsilon_1)\,d\epsilon_1.

The paper also defines the angular PSF (Eq. 37):

.. math::

   H^{2\nu}
   = \frac{\mathcal{A}_\theta}{\ln 2}\int_1^{T+1}\chi_{2\nu}(\epsilon_1)\,d\epsilon_1.

Implementation Notes
--------------------

SPADES maps these formulas to numerical objects as follows:

- ``\omega_{2\nu}``-type terms are evaluated in the ``Single`` and ``Sum`` spectra paths.
- ``\chi_{2\nu}``-type terms are evaluated in the ``Angular`` spectra path.
- ``\alpha(\epsilon_1)`` is represented by the ratio ``(dH/de)/(dG/de)``.
- 2nu channels support both ``Closure`` and ``Taylor`` approximations, matching the paper's treatment of neutrino potentials/denominators.

Channel-specific constants and integrands are implemented in:

- :mod:`spades.spectra.twobeta`
- :mod:`spades.spectra.ecbeta`
- :mod:`spades.spectra.twoec`
- :mod:`spades.spectra.closure_helpers`
- :mod:`spades.spectra.taylor_helpers`
