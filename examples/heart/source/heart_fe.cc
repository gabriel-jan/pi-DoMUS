#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/function_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/fe_field_function.h>

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "../include/heart_fe.h"

using namespace dealii;

/**
* In this class, the data is read completely and stored in
* a solution vector. A push forward function is provided to
* retrieve a heart point when inserting  cylinder point.
* In short:
* - creating a 2D geometry of the cylinder dimension
* - applying a finite element system on the mesh, one 
*   fe per coordinate
* - interpolate the data points at a fe degraa of 4 
* - differentiating between a hull point and abottom boundary
*   point
*/


template <int dim, int spacedim>
Heart<dim, spacedim>::Heart()
    : fe(FE_Q<dim>(2), spacedim)
    , dof_handler(triangulation)
{
}

template <int dim, int spacedim>
Heart<dim, spacedim>::Heart(bool side)
    : fe(FE_Q<dim>(4), spacedim)
    , dof_handler(triangulation)
    , solution(100)
    , side(side)
{
  if (side) {
    run_side();
  } else {
    run_bottom();
  }
}

template <int dim, int spacedim>
void Heart<dim, spacedim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  Point<dim> direction(1e-5, 1e5);
  // lexicographical numbering of the dofs
  // due to the heart point ordering
  DoFRenumbering::downstream(dof_handler, direction, true);
}

template <int dim, int spacedim>
void Heart<dim, spacedim>::reinit_data()
{
  std::string filename;
  // reading the correct data file
  if (side) {
    filename = "../source/side_boundary.txt";
  } else {
    filename = "../source/bottom_boundary.txt";
  }
  std::fstream in(filename);
  std::string coord;
  int n_dofs = dof_handler.n_dofs();

  for (int line = 0; line < 100; ++line) {
    std::getline(in, coord);

    // split into 3675 (side) or 507 (bottom) pieces
    std::vector<std::string> splitted;
    boost::split(splitted, coord, boost::is_any_of(";"));

    // write into solution
    solution[line].reinit(n_dofs);
    for (int column = 0; column < n_dofs; ++column) {
      solution[line][column] = std::stod(splitted[column]);
    }
  }
}

template <int dim, int spacedim>
Point<spacedim> Heart<dim, spacedim>::push_forward(const Point<dim> chartpoint,
                                                   const int timestep) const
{

  dealii::Functions::FEFieldFunction<dim, DoFHandler<dim>, Vector<double>>
      fe_field(dof_handler, solution[timestep]);
  Vector<double> wert(spacedim);
  fe_field.vector_value(chartpoint, wert);

  return Point<spacedim>(wert[0], wert[1], wert[2]);
}

template <int dim, int spacedim>
void Heart<dim, spacedim>::run_side()
{
  std::vector<unsigned int> subdivisions(2);
  subdivisions[0] = 48 / fe.degree;
  subdivisions[1] = 24 / fe.degree;
  const Point<dim> p1(0, -4.3196);
  const Point<dim> p2(2 * numbers::PI, 1.6838);

  GridGenerator::subdivided_hyper_rectangle(
      triangulation, subdivisions, p1, p2, false);
  setup_system();
  reinit_data();
}

template <int dim, int spacedim>
void Heart<dim, spacedim>::run_bottom()
{
  std::vector<unsigned int> subdivisions(2);
  subdivisions[0] = 12 / fe.degree;
  subdivisions[1] = 12 / fe.degree;
  const Point<dim> p1(-1.3858, -1.3858);
  const Point<dim> p2(1.3858, 1.3858);

  GridGenerator::subdivided_hyper_rectangle(
      triangulation, subdivisions, p1, p2, false);
  setup_system();
  reinit_data();
}

// Explicit instantiations
template class Heart<2, 3>;
