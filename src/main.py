from learn.linear_regression import (
  plot_scatter,
  ml_bp_disease_linear_regression,
  plot_learning_curve,
  plot_net
)
from learn.non_linear_regression import (
  do_ml_quadratic,
  do_ml_cos
)
from learn.relu import ( plot_relu )

def linear_regression():
  net, history = ml_bp_disease_linear_regression()
  plot_learning_curve(history)
  plot_net(net)

do_ml_cos()
