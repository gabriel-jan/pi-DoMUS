#include "boundary_values.h"
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#define PI 3.14159265358979323846

using namespace dealii;

// - - - - -  public functions - - - - -
// vector value is a virtual placehodler function to be overwritten easily
// It is templated with the dimension but it is not sure that the 3D case is 
// still working.
template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                       Vector<double> &values) const
{
  // if conditons concerning coordinates are to apply zero 
  // BC on the tube and the correct ones on the heart.
  // Pay attention, in 2D colour is colour-1 -> header file
  if (derivative)
      if (p[1]<=3.0014-1.318)
      {
        get_values_dt(p, values);
      }
      else
      {
        for (int i = 0; i < dim; ++i)
        {
          values[i]=0;
        }
      }
  else {
    if (color == 2) {
      get_heartdelta(p, values, heartstep);
    } else
      if (p[1]<=3.0014-1.318)
      {
        get_values(p, values);
      }
      else
      {
        for (int i = 0; i < dim; ++i)
        {
          values[i]=0;
        }
      }
  }
}

// - - - - -  private functions - - - - -
// rotate a certain point p by an angle "rot"
// returns also the height and the angle by reference
template <int dim>
Point<3> BoundaryValues<dim>::rotate(const Point<3> &p,
                                     double rot,
                                     double &angle,
                                     double &height) const
{
  // convert point to polar coordinates
  double x = p[0], y = p[1], z = p[2],
         phi = atan2(y, z); // returns angle in the range from -Pi to Pi

  // need to rotate to match heart_fe
  phi = phi + rot;

  // angle needs to be in the range from 0 to 2Pi
  phi = (phi < 0) ? phi + 2 * PI : phi;

  angle = phi;
  height = x;

  // transform back to cartesian
  double r = sqrt(y * y + z * z);
  z = r * cos(phi);
  y = r * sin(phi);

  return Point<3>(x, y, z);
}
// different coordinate systems have been used in the 
// team. confusion. 
template <int dim>
void BoundaryValues<dim>::swap_coord(Point<3> &p) const
{
  // swap x and z coordinate
  double tmp = p(0);
  p(0) = p(2);
  p(2) = tmp;
}
// creating a geometry in one dimension to apply a finite 
// element and interpolate the steps later.
template <int dim>
void BoundaryValues<dim>::setup_system()
{
  std::vector<unsigned int> subdivisions(1);
  subdivisions[0] = 1;
  const Point<1> p1(-1);
  const Point<1> p2(1);

  GridGenerator::subdivided_hyper_rectangle(
      triangulation, subdivisions, p1, p2, false);
  dof_handler.distribute_dofs(fe);
  Point<1> direction(1e5);
  DoFRenumbering::downstream(dof_handler, direction, true);
}

// retrieving the distance between a cylinder point and a heart point.
// equipping the 2D point with an artificial coordinate to fit 
// in the mapping function. The different faces are not aligned 
// correctly because of the data sets, so they are rotated and adjusted.
template <int dim>
void BoundaryValues<dim>::get_heartdelta(const Point<dim> point,
                                         Vector<double> &values,
                                         int step) const
{
  Point<3> rotated_p;
  Point<3> p;
  for (int i = 0; i < dim; ++i)
    p(i) = point(i);

  Point<3> artificial_p(point(1), 0, point(0));
  double rotation_offset = 0, rotate_slice = 0, phi, h;

  if (dim == 2) {
    rotation_offset = -PI / 4;
    // rotate the 2D slice by this angle
    rotate_slice = -PI /4;
    p = artificial_p;
  }

  Vector<double> rot(4);
  rot(0) = -PI / 4;
  rot(1) = -PI / 4;
  rot(2) = 0.0;
  rot(3) = PI / 4;

  rot.add(rotate_slice + rotation_offset);

  if (color == 2 && dim == 3) //////////////////////////////// top face
  {
    rotated_p = rotate(p, rot(color + 1), phi, h);

    // calc delta
    values(0) = 0;
    values(1) = rotated_p(1) - p(1);
    values(2) = rotated_p(2) - p(2);
  } else if (color == 1) //////////////////////////////// bottom face
  {
    Point<2> two_dim_pnt(p(2), p(1));

    if (dim == 2) {
      rotated_p = rotate(p, rot(color + 1), phi, h);
      two_dim_pnt(0) = rotated_p(2);
      two_dim_pnt(1) = rotated_p(1);
    }
    // get heart boundary point
    Point<3> heart_p = heart_bottom.push_forward(two_dim_pnt, step);
    swap_coord(heart_p);

    if (dim == 2) {
      rotated_p = rotate(heart_p, -rotate_slice, phi, h);
      heart_p = rotated_p;
    }

    // calculate delta
    int offset = (dim == 2) ? 2 : 0;
    for (int i = 0; i < dim; ++i) {
      int index = (i + offset) % 3;
      values(i) = heart_p(index) - p(index);
    }
    // The <=  0 thing is the reason 
    // of the colour = colour -1 thing
  } else if (color <= 0) //////////////////////////////// hull
  {
    rotated_p = rotate(p, rot(color + 1), phi, h);

    Point<2> polar_pnt(phi, h);

    // get heart boundary point
    Point<3> heart_p = heart_side.push_forward(polar_pnt, step);
    swap_coord(heart_p);

    if (dim == 2) {
      rotated_p = rotate(heart_p, -rotate_slice, phi, h);
      heart_p = rotated_p;
    }

    // calculate delta
    int offset = (dim == 2) ? 2 : 0;
    for (int i = 0; i < dim; ++i) {
      int index = (i + offset) % 3;
      values(i) = heart_p(index) - p(index);
    }
  }
}
// store three consecutive time steps to get an accurate interpolation 
// with the finite element.
template <int dim>
void BoundaryValues<dim>::get_values(const Point<dim> &p,
                                     Vector<double> &values) const
{
  int n_dofs = dof_handler.n_dofs();
  Vector<double> solution(n_dofs);

  std::vector<Vector<double>> u(3);
  u[0].reinit(dim);
  u[1].reinit(dim);
  u[2].reinit(dim);

  Point<1> substep(fmod(timestep, heartinterval) / heartinterval);

  int last = (heartstep == 0) ? 99 : heartstep - 1;
  int current = heartstep;
  int next = (heartstep == 99) ? 0 : heartstep + 1;

  BoundaryValues<dim>::get_heartdelta(p, u[0], last);
  BoundaryValues<dim>::get_heartdelta(p, u[1], current);
  BoundaryValues<dim>::get_heartdelta(p, u[2], next);

  int counter = 0;
  for (int line = 0; line < 3; ++line) {
    for (int column = 0; column < dim; ++column) {
      solution(counter) = u[line](column);
      ++counter;
    }
  }
  auto cell = GridTools::find_active_cell_around_point(dof_handler, substep);
  Point<1> scaled_point((substep(0) - cell->vertex(0)[0]) /
                        (cell->vertex(1)[0] - cell->vertex(0)[0]));

  Quadrature<1> quad(scaled_point);

  FEValues<1> fe_values(fe, quad, update_values);
  fe_values.reinit(cell);
  std::vector<Vector<double>> wert(quad.size(), Vector<double>(dim));
  fe_values.get_function_values(solution, wert);

  for (int i = 0; i < dim; ++i) {
    values(i) = wert[0](i);
  }
}
// calculates the velocity boundary conditions from the displacements 
// and then interpolates between two different velocity steps.
template <int dim>
void BoundaryValues<dim>::get_values_dt(const Point<dim> &p,
                                        Vector<double> &values) const
{
  Vector<double> u_minus(dim); // u_t-1
  Vector<double> u(dim);       // u_t
  Vector<double> u_plus(dim);  // u_t+1
  Vector<double> v0(dim);      // u_t - u_t-1
  Vector<double> v1(dim);      // u_t+1 - u_t
  Vector<double> delta_v(dim); // v1 - v0
  double substep = fmod(timestep, heartinterval) / heartinterval;

  int last = (heartstep == 0) ? 99 : heartstep - 1;
  int current = heartstep;
  int next = (heartstep == 99) ? 0 : heartstep + 1;

  BoundaryValues<dim>::get_heartdelta(p, u_minus, last);
  BoundaryValues<dim>::get_heartdelta(p, u, current);
  BoundaryValues<dim>::get_heartdelta(p, u_plus, next);

  // (u_t - u_t-1) / h
  v0 = u;
  v0 -= u_minus;
  v0 /= heartinterval;
  // (u_t+1 - u_t) / h
  v1 = u_plus;
  v1 -= u;
  v1 /= heartinterval;
  // calculate & scale delta_v
  delta_v = v1;
  delta_v -= v0;
  delta_v *= substep;

  v0 += delta_v; // delta_u = delta_u*substep

  // (wrong!) scaling to get CGS unit system
  // v0 /= 200;

  for (int i = 0; i < 2 * dim; ++i) {
    values(i) = v0(i % dim);
  }
}

// Explicit instantiations
template class BoundaryValues<2>;
template class BoundaryValues<3>;
