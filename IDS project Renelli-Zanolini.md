# IDS project Renelli-Zanolini

The project focuses on searching for missing persons in an avalanche scenario using the **ARTVA** signal transmitted by buried victims. The signal is detected by multiple robotic agents, mainly **quadcopters (drones)**, which act as receivers.

The first phase of the project consists of developing models for the main components of the system.

Each quadcopter measures two things, the time of return of a sonar pointed to the ground and the intensity of the ARTVA signal.

![drone_model](https://github.com/user-attachments/assets/eba4ee24-2e69-4a53-a723-b09b32d9f7c2)

---

## Single Drone Model

It's important to say that both estimates are noisy and **biased**.

### ARTVA signal

ARTVA devices operate at the standard frequency of **457 kHz (±80 Hz)** according to the European standard **ETSI EN 300 718**.

This frequency corresponds to a wavelength:

$$
\lambda = \frac{c}{f} \approx \frac{3 \cdot 10^8}{4.57 \cdot 10^5} \approx 656 \text{ m}
$$

which is **much larger than typical search distances (10–80 m).**

The main consequence is that the signal **does not behave like a classical radiative electromagnetic wave**, but mainly as a **magnetic field in the near-field of a dipole**.

In fact, the beacon uses:

- a **ferrite coil antenna**

- a **quasi-static oscillating magnetic field**

The **field lines** have the typical shape of a **magnetic dipole**, i.e., closed curves that originate from the ends of the antenna.

The magnetic field magnitude decreases approximately as:

$$
|B| \propto \frac{1}{\rho^3}
$$

When we measure the magnitude of this magnetic field we have to keep in mind that the distance is given by the Euclidean distance

$$
\rho = \sqrt{x^2 + y^2 + z^2}=\sqrt{X^2+z^2} = \sqrt{X^2+h_i^2}
$$

Where

- $X$ is the norm of the distance on the $(x,y)$ plane, the one we want to estimate

- The $h_i$ component depends on the altitude of the drone from the snow surface above the buried person $d_a$, and on the depth of buring $d_s$

$$
h_i = d_s+d_a
$$

The $h_i$ component acts as a bias, because for the same goal position on the map $(x,y)$ the signal will decrease when $h_i$ increases. The two components

- $d_s$ is fixed given the scenario.

- $d_a$ will change with the drone altitude and it is estimated through a measurement with a sonar under the drone $d_d$, that is intrinsically biased another time by unknown effects such as the mountain profile and the snow coverage after the avalanche. The bias on the quantity $d_d$ is not fixed, it depends on the drone position, and it decreases by getting closer to the goal location, in the limit $d_d = d_a$ when the drone is exactly above the buried person.

![height](https://github.com/user-attachments/assets/0ad41cef-68f6-45df-ace7-925f09665878)

### Bias and uncertainty on $X$ estimate

To provide a rigorous mathematical derivation for the bias and uncertainty of the horizontal distance $X$, we follow the methodology of **error propagation** through Taylor series expansion.

#### 1. The Functional Model

The relationship between the ARTVA signal intensity $|B|$, the vertical component $h$ (altitude + burial depth), and the horizontal distance $X$ is derived from the magnetic dipole near-field decay law:

$$
|B| = \frac{k}{\rho^3} = \frac{k}{(X^2 + h^2)^{3/2}}
$$

Solving for our variable of interest $X$, we define the function $g(|B|, h)$:

$$
X = g(|B|, h) = \sqrt{\left(\frac{k}{|B|}\right)^{2/3} - h^2}
$$

#### 2. The Measurement Model

Following the representation of measurement processes, we consider the observed quantities $z_B$ (signal intensity) and $z_h$ (estimated vertical component):

- **Intensity Measurement:** $z_B = |B|_a + \epsilon_B$, where $|B|_a$ is the actual value and $\epsilon_B \sim N(0, \sigma_B^2)$.
- **Vertical Estimate:** $z_h = h_a + b_h + \epsilon_h$, where $h_a$ is the actual depth/altitude, **$b_h$ is the systematic bias** introduced by the sonar/terrain, and $\epsilon_h \sim N(0, \sigma_h^2)$.

We aim to write the estimated value $\hat{X}$ as:

$$
\hat{X} = X_a + b_X + \epsilon_X
$$

#### 3. Taylor Series Expansion for Uncertainty and Bias

To retrieve the components $b_X$ and $\epsilon_X$, we perform a Taylor expansion of $g(|B|, h)$ around the actual values $(|B|_a, h_a)$:

$$
\hat{X} \approx g(|B|_a, h_a) + \left. \frac{\partial g}{\partial |B|} \right|_a (z_B - |B|_a) + \left. \frac{\partial g}{\partial h} \right|_a (z_h - h_a) + \dots
$$

Where $g(|B|_a, h_a) = X_a$.

##### A. The Jacobian (First-Order Derivatives)

We calculate the sensitivity of $X$ with respect to its inputs:

1. **Sensitivity to Intensity:** $\frac{\partial g}{\partial |B|} = -\frac{(k/|B|)^{2/3}}{3X|B|} = -\frac{\rho^2}{3X|B|} = -\frac{\rho^5}{3Xk}$.
2. **Sensitivity to Height:** $\frac{\partial g}{\partial h} = -\frac{h}{X}$.

##### B. Deriving the Noise Term ($\epsilon_X$)

The random noise component $\epsilon_X$ is the part of the expansion associated with the zero-mean random variables $\epsilon_B$ and $\epsilon_h$:

$$
\epsilon_X \approx \left( -\frac{\rho^5}{3Xk} \right) \epsilon_B + \left( -\frac{h}{X} \right) \epsilon_h
$$

The **uncertainty** (standard deviation $\sigma_X$) is found by calculating the variance:

$$
\sigma_X = \sqrt{ \left( \frac{\partial g}{\partial |B|} \right)^2 \sigma_B^2 + \left( \frac{\partial g}{\partial h} \right)^2 \sigma_h^2 } = \frac{1}{X} \sqrt{ \left( \frac{\rho^5}{3k} \right)^2 \sigma_B^2 + h^2 \sigma_h^2 }
$$

*Note: As $X \to 0$ (drone directly above the victim), the uncertainty $\sigma_X$ grows unbounded, confirming that horizontal precision degrades at the vertical.*

##### C. Deriving the Bias Term ($b_X$)

The bias consists of two contributions:

**Systematic Propagation:** The propagation of the sonar bias $b_h$:

$$
b_{X, \text{sys}} \approx \frac{\partial g}{\partial h} b_h = -\frac{h}{X} b_h
$$

**Non-linearity Bias:** Since $g$ is a curved function, even if $\epsilon$ is zero-mean, the expected value $E[g(z)]$ does not equal $g(E[z])$. This is captured by second-order terms in the Taylor expansion:

$$
b_{X, \text{non-lin}} \approx \frac{1}{2} \left( \frac{\partial^2 g}{\partial |B|^2} \sigma_B^2 + \frac{\partial^2 g}{\partial h^2} \sigma_h^2 \right)
$$

Total Bias:

$$
b_X = -\frac{h}{X} b_h + \frac{1}{2} \left( \frac{\partial^2 g}{\partial |B|^2} \sigma_B^2 + \frac{\partial^2 g}{\partial h^2} \sigma_h^2 \right)
$$

### Summary of Results

The rigorous mathematical model for the estimated horizontal distance is:

$$
\hat{X} = X_a + \underbrace{\left[ -\frac{h}{X} b_h + \text{Higher Order Terms} \right]}_{\text{Bias } b_X} + \underbrace{\left[ -\frac{\rho^5}{3Xk} \epsilon_B - \frac{h}{X} \epsilon_h \right]}_{\text{Noise } \epsilon_X}
$$

**Some notes:**

- **Bias Definition:** Bias is the systematic deviation $E[\hat{X}] - X_a$, caused here by both sonar inaccuracy ($b_h$) and the "curvature" of the dipole field.
- **Jacobian Importance:** The Jacobian terms ($1/X$) show that the coordinate transformation from the magnetic field to the horizontal plane is inherently unstable near the origin.

This is how a single drone estimates the distance from the victim, so it gets a "circle", but to estimate $X$ position we need 2 or 3 drones.

---

## Multi-agent Model

The following model extends the single-drone scenario to a collaborative multi-agent system (2 or 3 drones) that utilizes relative distance measurements (UWB) and distributed estimation algorithms to localize the victim.

![multi-agent](https://github.com/user-attachments/assets/7f7589da-5762-4902-9336-d0c925bd9843)


### 1. Multi-Agent State Definition

In the single-drone model, we estimated a scalar horizontal distance $X$. To estimate a specific 2D position $(x_T, y_T)$ for the buried victim, we define the global state as the target position $x_T$ and the relative positions of the drones $p_i$.

Since GPS is unavailable, the drones establish a **Local Coordinate System**. One drone is designated as the origin $(0,0)$, and the others define their positions relative to it using inter-drone measurements.

### 2. The Measurement Equations

Each drone $i$ collects two types of measurements:

1. **Local Measurements (ARTVA & Sonar):** As in the single drone model, the intensity $|B|_i$ follows the dipole decay law $|B|_i = \frac{k}{\rho_i^3}$. The estimated horizontal distance for drone $i$ is:

$$
\hat{X}_i = \sqrt{\left(\frac{k}{|B|_i}\right)^{2/3} - (d_{a,i} + d_s + b_{d,i})^2}
$$

        where $d_{a,i}$ is altitude, $d_s$ is burial depth, and $b_{d,i}$ is the sonar bias.

2. **Relative Measurements (UWB/TOF):** To link the drones' frames, they measure the relative distance $d_{ij}$ using **Ultra-Wide Band (UWB)** technology based on **Time of Flight (TOF)**. This provides a precise range $d_{ij} = |p_j - p_i| + \eta_{ij}$ that is resistant to multipath interference in snowy terrain.

### 3. Distributed Estimation Framework

To avoid a single point of failure and reduce communication overhead, the drones use a **Distributed Weighted Least Squares (WLS)** approach based on **Linear Consensus**.

1. **Local Initialisation:** Each drone $i$ computes a local information matrix $F_i(0)$ and an information vector $a_i(0)$:
   - $F_i(0) = H_i^T R_i^{-1} H_i$ (representing the precision of drone $i$’s measurement).
   - $a_i(0) = H_i^T R_i^{-1} z_i$ (representing the direction/distance to the target).
2. **Consensus Rounds:** The drones exchange these values with neighbors. Through iterative updates, they reach an agreement on the global sum of information:

$$
F_i(k+1) = \sum_{j=1}^n q_{ij} F_j(k), \quad a_i(k+1) = \sum_{j=1}^n q_{ij} a_j(k)
$$

    Using **Metropolis-Hastings weights** ($q_{ij}$) ensures the fastest convergence to the     global average.

### 4. Bias Removal Mechanism

The model removes bias through two primary methods:

- **Spatial Diversity:** The single drone model notes that sonar bias $b_d$ decreases as the drone approaches the vertical of the victim ($X \to 0$). By having multiple drones, the drone closest to the victim can serve as a **reference system** with negligible systematic effects. The differences between drones' estimates of $X$ allow the system to solve for the individual biases $b_{d,i}$ using **Polynomial Regression**.
- **Redundancy in $d_s$:** The burial depth $d_s$ is a common constant for all drones. In a multi-drone Least Squares setup, $d_s$ is treated as an unknown parameter in the information vector $a(k)$. The redundant measurements from different angles allow the estimator to decouple $d_s$ from the random noise $\epsilon$ and individual sonar biases $b_d$.

### 5. Final Position Estimate

After consensus is reached, each drone calculates the optimal, unbiased target position:

$$
\hat{x}_T = F_{global}^{-1} a_{global}
$$

The resulting covariance $P$ is significantly smaller than that of a single drone, as the Fisher Information grows linearly with the number of sensors. This collaborative approach turns the individual "circles" of distance into a precise coordinate $(x_T, y_T)$.
