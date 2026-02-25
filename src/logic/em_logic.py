import numpy as np
import scipy.stats
import cv2
import pandas as pd

class GMM:
    def __init__(self, k, max_iter=5, variances=None):
        self.k = k
        self.max_iter = max_iter
        if variances is None:
            self.variances = np.ones(self.k)
        else:
            self.variances = variances

    def initialize(self, X):
        rows, cols = X.shape
        self.cols = cols
        self.rows = rows
        indexes = np.arange(rows)
        np.random.shuffle(indexes)
        self.mu = X[indexes[:self.k]]
        self.sigma = np.array([np.eye(cols) * self.variances[i] for i in range(self.k)])

    def e_step(self, X):
        # E-Step: 
        self.weights = self.predict_proba(X) # calculate the fraction of points belonging to each cluster
        self.phi = np.mean(self.weights, axis=0)

    def m_step(self, X, fix_variance=False, perc_thresh=80):
        # M-Step: 
        for class_idx in range(self.k):
            w = self.weights[:, class_idx].copy() # weight assigned to each data point for the class
            if w.sum() == 0:
                self.sigma[class_idx] = np.identity(self.sigma[class_idx].shape[0]) * 0.00001
                self.mu[class_idx] = (self.mu[class_idx] * 0) 
                continue
            
            probs = w / w.sum() # probability of each data point belonging to the class
            lower_percentile = np.percentile(probs, perc_thresh)
            # set the weights of the data points assigned to the class to 0 for subsequent clusters in this M-step
            self.weights[:, class_idx+1:][(probs >= lower_percentile)] = 0 
            
            self.mu[class_idx] = probs @ X    # mean of the data points assigned to the class
            
            if not fix_variance:
                diff = X - self.mu[class_idx]
                self.sigma[class_idx] = (probs * diff.T) @ diff  # covariance matrix
                vv = self.sigma[class_idx].copy()
                vv_det = abs(np.linalg.det(vv))
                vv_side = (vv_det / np.pi)**(1.0 / self.cols)
                self.sigma[class_idx] = np.diag([vv_side] * self.cols)

    def fit(self, X, fix_variance=False, perc_thresh=80):
        self.initialize(X)  # sets the means and covariance matrices
        for _ in range(self.max_iter):
            self.e_step(X)      # E-step
            self.m_step(X, fix_variance, perc_thresh)      # M-step

    def predict_proba(self, X):
        # Prevent singular matrix errors
        eps = 1e-6
        Px_Pk = []
        for i in range(self.k):
            # Ensure sigma is positive definite
            cov = self.sigma[i] + np.eye(self.cols) * eps
            try:
                prob = scipy.stats.multivariate_normal.pdf(X, mean=self.mu[i], cov=cov)
            except:
                prob = np.zeros(X.shape[0])
            Px_Pk.append(prob)
        Px_Pk = np.array(Px_Pk).T
        
        row_sums = Px_Pk.sum(axis=1)[:, np.newaxis]
        return Px_Pk / (row_sums + eps)

def run_em_optimization(target_mask, radii_m, map_width_m, max_iter=20, update_radii=False, perc_thresh=80):
    """
    Optimizes sensor placement using an EM algorithm (GMM).
    Yields intermediate results for streaming.
    """
    h, w = target_mask.shape
    scale = w / map_width_m
    
    target_points = np.argwhere(target_mask > 0)
    if len(target_points) == 0:
        return

    if len(target_points) > 1000:
        indices = np.random.choice(len(target_points), 1000, replace=False)
        target_points = target_points[indices]

    variances_px = [( (r * scale) / 3 )**2 for r in radii_m]
    
    gmm = GMM(k=len(radii_m), max_iter=max_iter, variances=variances_px)
    gmm.initialize(target_points)
    
    for i in range(max_iter):
        gmm.e_step(target_points)
        gmm.m_step(target_points, fix_variance=not update_radii, perc_thresh=perc_thresh)
        
        # Calculate current state for yielding
        means = gmm.mu # [y, x]
        covs = gmm.sigma
        
        current_radii_m = []
        for j in range(len(radii_m)):
            v = covs[j][0, 0]
            r_m = (3 * (v**0.5)) / scale
            current_radii_m.append(r_m)
            
        config = []
        current_iter_mask = np.zeros((h, w), dtype=np.uint8)
        centroid = np.mean(target_points, axis=0)
        
        for idx in range(len(means)):
            y, x = means[idx]
            angle = np.degrees(np.arctan2(-y + centroid[0], x - centroid[1]))
            r_m = current_radii_m[idx]
            
            # Update mask
            r_px = int(r_m * scale)
            cv2.circle(current_iter_mask, (int(x), int(y)), r_px, 255, -1)
            
            config.append({
                'x_m': round(x / scale, 2),
                'y_m': round(y / scale, 2),
                'angle_deg': round(angle, 1),
                'radius_m': round(r_m, 1)
            })
        
        intersection = np.bitwise_and(current_iter_mask, target_mask)
        coverage_pct = (np.sum(intersection > 0) / np.sum(target_mask > 0)) * 100
        results_df = pd.DataFrame(config)
        
        yield current_iter_mask, coverage_pct, results_df[['x_m', 'y_m', 'angle_deg', 'radius_m']], i + 1

