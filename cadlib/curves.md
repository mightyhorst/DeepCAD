The purpose of `/content/code/cadlib/curves.py` is to provide classes and functions for working with curves in CAD models. It includes functionality for creating, manipulating, and sampling curves.

Here is a code snippet from `/content/code/cadlib/curves.py` that shows the `sample_points` function:

```python
def sample_points(self, n=32):
    points = np.stack([curve.sample_points(n) for curve in self.children], axis=0) # (n_curves, n, 2)
    return points
```

This function takes an optional argument `n` for the number of points to sample on each curve. It iterates over each curve in the `children` list and calls the `sample_points` function on each curve to generate the sampled points. The sampled points are then stacked together into a numpy array with shape `(n_curves, n, 2)`, where `n_curves` is the number of curves and `n` is the number of sampled points on each curve. Finally, the function returns the sampled points.