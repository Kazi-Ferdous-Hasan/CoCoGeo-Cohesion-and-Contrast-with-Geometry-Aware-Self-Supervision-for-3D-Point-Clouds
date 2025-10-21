# CoCoGeo-Cohesion-and-Contrast-with-Geometry-Aware-Self-Supervision-for-3D-Point-Clouds
# CoCoGeo: Cohesion and Contrast with Geometry-Aware Self-Supervision for 3D Point Clouds

This repository implements **CoCoGeo**, a self-supervised learning (SSL) framework that learns robust, geometry-aware representations from **unlabeled 3D point clouds**, and transfers them to a downstream **YOLO3D** detector for object detection.  
The pipeline follows:  
**Self-Supervised Pretraining → Save Backbone → Load Backbone in YOLO3D → Fine-tune → Evaluate.**

---

## 🔹 1. Geometric Preprocessing and Augmentation

**Input:** Raw 3D point cloud of \( N \) points with coordinates \((x, y, z)\).  
We randomly sample \( N = 4096 \) points per frame.

### Augmentations
To enforce invariance to scale, rotation, and noise:
- **Rotation (around Z-axis)**:
  \[
  R_z(\alpha) =
  \begin{bmatrix}
  \cos\alpha & -\sin\alpha & 0 \\
  \sin\alpha & \cos\alpha & 0 \\
  0 & 0 & 1
  \end{bmatrix}, \quad p' = R_z(\alpha)p
  \]
- **Scaling:** multiply all coordinates by a random factor \( s \in [0.9, 1.1] \)
- **Jitter:** add Gaussian noise (\(\sigma \approx 0.01\))

These augmentations preserve object shape while promoting orientation and scale robustness.

---

### Bird’s-Eye-View (BEV) IoU
For boxes with center \((x, y)\), dimensions \((w, l)\), and yaw \( \theta \), the 2D polygon IoU is:
\[
IoU(P_1, P_2) = \frac{\text{area}(P_1 \cap P_2)}{\text{area}(P_1 \cup P_2)}
\]
For axis-aligned boxes, intersection is computed from X and Y intervals.

### Yaw Loss
Object orientation is optimized using a cosine periodic loss:
\[
\mathcal{L}_{yaw}(\hat{\theta}, \theta) = 1 - \cos(\hat{\theta} - \theta)
\]

---

## 🔹 2. Self-Supervised Pretext Task (Point-CoCo)

### Overview
CoCoGeo pretrains a **PointNet-style encoder** via three complementary objectives:
1. **Masked Reconstruction (Chamfer loss)**
2. **Contrastive Regularization (BYOL)**
3. **Geometry Consistency (Eigenvalue-based classification)**

---

### 2.1 Point Encoder (Backbone)

Each input point \( p_i \in \mathbb{R}^3 \) is encoded via:
\[
f_i = BN(ReLU(W_2 \, ReLU(W_1 p_i)))
\]
Output: feature tensor \( F \in \mathbb{R}^{N \times 128} \)

---

### 2.2 Masking and Reconstruction

- Randomly mask 75% of input points.
- Encode remaining subset \( M \subset \{1,\dots,N\} \)
- Aggregate global feature:
  \[
  g = \max_i f_i
  \]
- Decode to predict all \( N \) points.

**Chamfer Distance Loss:**
\[
d_{CD}(P, Q) = 
\frac{1}{|P|}\sum_{p \in P}\min_{q \in Q}\|p - q\|^2
+ \frac{1}{|Q|}\sum_{q \in Q}\min_{p \in P}\|p - q\|^2
\]

---

### 2.3 Contrastive (BYOL) Loss

Two augmented versions are processed through:
- **Online network** (with predictor)
- **Target network** (EMA-updated backbone)

Loss:
\[
\mathcal{L}_{BYOL} = 2 - 2\cdot\frac{p^\top z}{\|p\|\,\|z\|}
\]
The target weights are updated via EMA:
\[
\theta_{target} \leftarrow \tau \theta_{target} + (1 - \tau)\theta_{online}, \quad \tau = 0.996
\]

---

### 2.4 Geometric Consistency Loss

For each unmasked point:
- Compute local covariance via k-NN.
- Use eigenvalues \(\lambda_1 \ge \lambda_2 \ge \lambda_3\) to label shape:
  - Linear, Planar, or Scattered
- Predict category via a small MLP classifier.

Loss:
\[
\mathcal{L}_{geo} = -\sum_i \log p(y_i)
\]

---

### 2.5 Total SSL Objective

\[
\boxed{
\mathcal{L}_{SSL} = \mathcal{L}_{BYOL} + \lambda_{CD}\, d_{CD} + \lambda_{geo}\, \mathcal{L}_{geo}
}
\]

All modules (encoder, projector, predictor, decoder, geo head) are trained jointly.  
After convergence, the **encoder weights are saved** as the pretrained backbone.

---

## 🔹 3. Downstream Task: 3D Object Detection (YOLO3D)

### 3.1 Architecture

- The **YOLO3D detector** reuses the pretrained encoder as backbone.
- Each point feature \( f_i \) is passed through a small MLP detection head outputting:
  \[
  [\Delta x, \Delta y, \Delta z, w, l, h, \theta, o, p_1,\dots,p_C]
  \]
  where:
  - \(o\): objectness score
  - \(p_1..p_C\): class logits
  - \(C\): number of object classes

---

### 3.2 Detection Losses

For assigned positive point predictions \( \hat{b} \) vs ground truth \( g \):

- **Regression Loss:**
  \[
  \mathcal{L}_{reg} = SmoothL1((\hat{x},\hat{y},\hat{z},\hat{w},\hat{l},\hat{h}) - (x,y,z,w,l,h))
  \]
- **Yaw Loss:**
  \[
  \mathcal{L}_{yaw} = 1 - \cos(\hat{\theta} - \theta)
  \]
- **Objectness Loss:** Binary cross-entropy on \( \hat{o} \)
- **Classification Loss:** Cross-entropy on predicted class \( \hat{p} \)

**Total Detection Loss:** weighted sum of the above terms.

---

### 3.3 Loading Pretrained Weights

The pretrained SSL backbone is seamlessly transferred:
```python
model = YOLO3D()
load_ssl_weights(model.backbone, 'pointcoco_pretrained.pth')
