---
title: "Review of *Multigroup First Crack* (26 January 2026)"
geometry: margin=1in
---

Equation numbers refer to the original document. Items are ordered by importance within each section.

# Variable Basis Method (Section 3)

## 1. Eq. (42) is incorrect (main error)

The text says $T$ cancels from both sides, but the left and right sides were scaled by different factors. Multiplying (41) by $T$ gives

$$
\frac{1}{c}\left(\frac{\partial\hat\psi}{\partial t} - \frac{u}{T}\frac{\partial T}{\partial t}\frac{\partial\hat\psi}{\partial u} - \frac{\hat\psi}{T}\frac{\partial T}{\partial t}\right) + \Omega_x\left(\frac{\partial\hat\psi}{\partial x} - \frac{u}{T}\frac{\partial T}{\partial x}\frac{\partial\hat\psi}{\partial u} - \frac{\hat\psi}{T}\frac{\partial T}{\partial x}\right) = 4\pi\sigma_a\hat B - \sigma_a\hat\psi .
$$

The left side of (42) as written corresponds to multiplying (41) by $T^2$, but the right side was only multiplied by $T$. A units check confirms the mismatch: the two sides of (42) differ by a factor of $T$. If the left side of (42) is kept as written, the right side must be

$$
T\left(4\pi\sigma_a\hat B - \sigma_a\hat\psi\right).
$$

This error carries into Eqs. (43) and (45), whose right sides should be $cT\sigma_a\left(4\pi\hat B - \hat\psi\right)$.

## 2. Recommended conservative form

Using $u\,\partial_u\hat\psi + \hat\psi = \partial_u(u\hat\psi)$, the two $\hat V$ terms combine. After dividing by $T$:

$$
\frac{1}{c}\frac{\partial\hat\psi}{\partial t} + \Omega_x\frac{\partial\hat\psi}{\partial x} - \frac{\hat V}{cT}\frac{\partial (u\hat\psi)}{\partial u} = \sigma_a\left(4\pi\hat B - \hat\psi\right),
\qquad \hat V = \frac{\partial T}{\partial t} + c\,\Omega_x\frac{\partial T}{\partial x}.
$$

This can be checked directly. Let $L = \frac{1}{c}\partial_t + \Omega_x\partial_x$ at fixed $\nu$. Then

$$
L\!\left(\frac{\hat\psi}{T}\right) = \frac{1}{T}\left[L\hat\psi\big|_u - \frac{LT}{T}\,\frac{\partial(u\hat\psi)}{\partial u}\right],
$$

with $LT = \hat V/c$.

The divergence form in $u$ is the right starting point for a group discretization in $u$, because it conserves energy across groups. Some consequences:

- $\hat V$ depends on angle through $\Omega_x$.
- The sign of $\hat V$ sets the upwind direction in $u$.
- $u\hat\psi \to 0$ at $u = 0$, so no boundary condition is needed there.

## 3. $\hat\psi$ must depend on $x$ and $\Omega$

Eqs. (32)–(45) write $\hat\psi(u,t)$. If that were literally true, $\partial\hat\psi/\partial x$ would vanish. It should be $\hat\psi(u,x,t,\Omega)$.

## 4. Notation in Eqs. (37)–(38)

As written, $\partial_t\hat\psi = \partial_t\hat\psi + \partial_u\hat\psi\,\partial_t u$ implies the last term is zero. The two derivatives must be distinguished: the left side is taken at fixed $\nu$, the right side at fixed $u$. For example,

$$
\left.\frac{\partial\hat\psi}{\partial t}\right|_\nu = \left.\frac{\partial\hat\psi}{\partial t}\right|_u + \frac{\partial\hat\psi}{\partial u}\left.\frac{\partial u}{\partial t}\right|_\nu,
\qquad \left.\frac{\partial u}{\partial t}\right|_\nu = -\frac{u}{T}\frac{\partial T}{\partial t}.
$$

With this interpretation, Eqs. (35)–(41) are correct.

## 5. Eq. (33) uses the wrong symbol

It should read $\psi(\nu,x,t) = \hat\psi(u,x,t)/T(x,t)$. This is the angular intensity $\psi$, not the scalar flux $\phi$.

## 6. Define $\hat B$ explicitly

From $\hat B\,du = B\,d\nu$:

$$
\hat B(u,T) = T\,B(uT,T) = T^4\,\frac{2hu^3/c^2}{e^{hu/k}-1}.
$$

This shows the motivation for the method: in $u$ the Planck spectrum has a fixed shape that only scales as $T^4$. Note also that the opacity becomes $\sigma_a(uT,T)$.

## 7. Eq. (46) mixes variables

It integrates over $\nu$ but uses $\hat\phi(u)$ and the group quantity $\bar B_g$. Using $\phi\,d\nu = \hat\phi\,du$, it should read

$$
C_v\frac{\partial T}{\partial t} = \int_0^\infty du\,\sigma_a(uT,T)\left[\hat\phi(u,x,t) - 4\pi\hat B(u,T)\right],
\qquad \hat\phi = \int \hat\psi\,d\Omega .
$$

Also, $T(t)$ should be $T(x,t)$.

## 8. Practical caveat

$u = \nu/T$ is singular at $T = 0$, and $\hat V$ contains $\partial_x T$, which is very large at a Marshak wave front. A cold initial condition will need a temperature floor, and the front will need care in the discretization.

# Gray and Multigroup Formulation (Sections 1–2)

## 9. The $4\pi$ factors are inconsistent

Eq. (1) has $4\pi\sigma_a B$, while Eq. (3) has $\sigma_a B$. The material equation (2), with $\phi = \int\psi\,d\Omega$, is consistent with $B$ being the Planck *intensity*. In that case the transport emission source should be $\sigma_a B$, with no $4\pi$. One convention should be chosen and applied to Eqs. (1), (3), (5), (14), (18), (25), (28), (34), and (45).

## 10. Eq. (15) has an extra $4\pi$

Since $\int_0^\infty B\,d\nu = acT^4/(4\pi)$, the emission term is

$$
4\pi\int_0^\infty \sigma_a B\,d\nu = \sigma_P\,acT^4,
$$

not $4\pi\sigma_P\,acT^4$. The same correction applies to Eq. (16).

## 11. Eq. (10) uses the wrong cross section

It has $\sigma_a$ where it should have $\sigma^*$. Eq. (11) correctly uses $\sigma^*$.

## 12. Sign error for $\mu_l < 0$ in Eqs. (12) and (30)

Starting from the upwind difference (9):

$$
\mu_l\frac{\psi_{i+1}-\psi_i}{\Delta x} + \sigma^*\psi_i = Q^*
\;\Longrightarrow\;
\psi_i = \frac{Q^* - \mu_l\psi_{i+1}/\Delta x}{\sigma^* - \mu_l/\Delta x}
= \frac{Q^* + |\mu_l|\,\psi_{i+1}/\Delta x}{\sigma^* + |\mu_l|/\Delta x}.
$$

As written, the denominator $\mu_l/\Delta x + \sigma^*$ can be zero or negative.

## 13. Eq. (22) is not an average

$\int\sigma_a\,d\nu$ has units of opacity × frequency. For Eq. (24) and the emission term to make sense, $\bar\sigma_{ag}$ must be a Planck-weighted average:

$$
\bar\sigma_{ag} = \frac{\int_{\nu_{g-1/2}}^{\nu_{g+1/2}} \sigma_a B\,d\nu}{\int_{\nu_{g-1/2}}^{\nu_{g+1/2}} B\,d\nu}.
$$

With this definition, $\int_g\sigma_a B\,d\nu = \bar\sigma_{ag}\bar B_g$ holds exactly, and Eq. (24) is an approximation. Also, the group bounds are written $\nu_{i\pm1/2}$, which clashes with the spatial index $i$; they should be $\nu_{g\pm1/2}$.

## 14. Eq. (28) is incorrect

It includes $-\bar\sigma_{ag}\bar\psi_g$, which is already in $\sigma^*$, so absorption is counted twice. It also drops the old-time-level term. It should be

$$
Q^*_{i,g,l} = 4\pi\bar\sigma_{ag}\bar B_g(T^n_i) + \frac{\bar\psi^{\,n,l}_{i,g}}{c\Delta t}
$$

(with the $4\pi$ convention fixed per item 9).

## 15. Eq. (31) treats emission explicitly

Eq. (31) evaluates emission at $B(T^n)$. This conserves energy, because the radiation emits the same amount the material loses. However, it can oscillate or become unstable when $\Delta t$ is large compared with $C_v/(4\sigma_P acT^3)$. The linearization in Eq. (17) is introduced but never used to close the system, for example through a Fleck factor or effective-scattering formulation.

# Minor Notation

- Eqs. (8)–(9): the subscript on $\mu$ should be $l$, not $i$.
- Eq. (13): stray comma in $\psi_{i+1/2,}$.
- Eq. (14): the superscript $l$ on $\sigma_a$ and $B$ denotes angle, but these are angle-independent; probably the group index was intended.
- Eq. (17): mixed sub/superscripts on $n$; should be $(T^4)^{n+1} \approx (T^n)^4 + 4(T^n)^3(T^{n+1}-T^n)$.
- Eq. (20): the "=" sits inside the integral's parentheses.
- Eq. (27): $\bar\phi(\nu,t,x)$ should be $\bar\phi_g(x,t)$.
- Eqs. (2), (19), (46): $T(t)$ should be $T(x,t)$.
- Eq. (7): calling it the "steady state eq" is misleading; it is the problem solved within each time step.

# Figure 1 (Reed's Problem)

Reed's problem includes scattering regions, but none of the equations in the document have a scattering term. The text also says the test problems used a constant $Q^*$. The figure therefore tests only the transport sweep, not the material coupling. The document should explain how scattering was handled.
