.. role:: raw-math(raw)
    :format: latex html

standard_fit
============

名前はてきとう

Linear Regression
~~~~~~~~~~~~~~~~~

y = 2x + 1
^^^^^^^^^^^^^^

.. code:: python

    import numpy as np
    import plotly.express as px
    import standard_fit.plotly.express as sfpx

    x = np.linspace(-10, 10, 100)
    y_sigma = [2] * 100
    y = -2 * x + 1 + np.random.normal(0, y_sigma)
    fig = px.scatter(x=x, y=y, error_y=y_sigma)
    sfpx.fit(fig, "pol1")
    fig.show()


.. image:: ./pol1.png

y = 2sin(x) + 5cos(2x) + 9cos(5x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    import numpy as np
    import plotly.express as px
    import standard_fit.plotly.express as sfpx

    x = np.linspace(-10, 10, 1000)
    y_sigma = [2] * len(x)
    y = 2 * np.sin(x) + 5 * np.cos(2 * x) + 9 * np.cos(5 * x) + np.random.normal(0, y_sigma)

    fig = px.scatter(title="Fourier-series fitting", x=x, y=y, error_y=y_sigma)
    sfpx.fit(fig, "fourier5")
    fig.update_xaxes(range=(-12, 22))
    fig.show()

    fig = px.scatter(title="Fourier-series fitting with LASSO regularization (λ=0.1)", x=x, y=y, error_y=y_sigma)
    sfpx.fit(fig, "fourier5", fit_kwargs=dict(lasso_lambda=0.1))
    fig.update_xaxes(range=(-12, 22))
    fig.show()

.. list-table::

    * - .. figure:: ./fs.png
    
      - .. figure:: ./fs_lasso.png

Note that error values cannot be calculated in linear regression with LASSO regularization
