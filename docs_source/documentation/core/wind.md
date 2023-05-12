# Wind

## Overview

PyFlyt supports various wind field models, as well as provides the capability for users to define their own wind field models.

## Preimplemented Wind Models

Several popular wind models are provided:

1. Lorem Ipsum

They can be initialized in the `aviary` like so:

```python
env = Aviary(..., wind_type="Lorem Ipsum")
```

## Custom Wind Modelling

To define custom wind models, refer the the example provided by the `WindFieldClass` below:

```{eval-rst}
.. autoclass:: PyFlyt.core.abstractions.WindFieldClass
```

### Default Attributes
```{eval-rst}
.. property:: PyFlyt.core.abstractions.WindFieldClass.np_random

  **dtype** - `np.random.RandomState`
```

### Required Methods
```{eval-rst}
.. autofunction:: PyFlyt.core.abstractions.WindFieldClass.__init__
.. autofunction:: PyFlyt.core.abstractions.DroneClass.__call__
```