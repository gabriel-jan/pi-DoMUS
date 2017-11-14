/*! \addtogroup equations
 * @{
 */

/**
 * This interface solves the ALE Navier Stokes problem:
 *
 */

#ifndef _pidomus_ALE_navier_stokes_h_
#define _pidomus_ALE_navier_stokes_h_

#include <boundary_values.h>
#include <pde_system_interface.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal2lkit/parsed_mapped_functions.h>
#include <deal2lkit/parsed_preconditioner/amg.h>
#include <deal2lkit/parsed_preconditioner/jacobi.h>

////////////////////////////////////////////////////////////////////////////////
/// ALE Navier Stokes interface:

template <int dim, int spacedim = dim, typename LAC = LATrilinos>
class ALENavierStokes
    : public PDESystemInterface<dim, spacedim,
                                ALENavierStokes<dim, spacedim, LAC>, LAC> {

public:
  virtual ~ALENavierStokes() {
    mapped_mapping = nullptr;
  }
  ALENavierStokes();

  void declare_parameters(ParameterHandler &prm);
  void parse_parameters_call_back();

  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      FEValuesCache<dim, spacedim> &scratch, std::vector<EnergyType> &energies,
      std::vector<std::vector<ResidualType>> &residuals,
      bool compute_only_system_terms) const;

  void compute_system_operators(
      const std::vector<shared_ptr<LATrilinos::BlockMatrix>>,
      LinearOperator<LATrilinos::VectorType> &,
      LinearOperator<LATrilinos::VectorType> &,
      LinearOperator<LATrilinos::VectorType> &) const;

  // overwriting empty placeholder signal functions 
  virtual void connect_to_signals() const {
    auto &signals = this->get_signals();
    disable_heart = this->get_disable_heart_bool();
    if (!disable_heart) {
      // The input mesh file needs to be coloured to apply boundary conditions 
      // properly. Then, the geometry is shifted to match the shape of the heart 
      // as close as possible to avoid large deformations of cells.
      signals.postprocess_newly_created_triangulation.connect(
          [&, this](Triangulation<dim, spacedim> *tria) {
            // manual coloring
            for(auto cell : tria->active_cell_iterators())
            {
              for(unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              {
                if(cell->face(f)->at_boundary())
                {
                  if(cell->face(f)->center()[0] <= -1.3857)
                  {
                    cell->face(f)->set_boundary_id(0);
                  }
                  if(cell->face(f)->center()[0] >= 1.3857)
                  {
                    cell->face(f)->set_boundary_id(1);
                  }
                  if(cell->face(f)->center()[1] <= -3.0014)
                  {
                    cell->face(f)->set_boundary_id(2);
                  }
                  // this value needs to be 3.0014 if the short mesh is used
                  if(cell->face(f)->center()[1] >= 5.0014) 
                  {
                    cell->face(f)->set_boundary_id(3);
                  }
                }
              }
            }
            // shifting the geometry to the right position
            int index = (dim == 2) ? 1 : 0;
            Tensor<1, dim> shift_vec;
            shift_vec[index] = -1.318;
            GridTools::shift(shift_vec, *tria);
          });
      // In the following signal, the boundary conditions are applied
      // to all boundary IDs. It is necessary to apply the deformation, 
      // the vleocity and also the time derivatives of the deformation.
      signals.update_constraint_matrices.connect(
          [&, this](std::vector<std::shared_ptr<dealii::ConstraintMatrix>>
                        &constraints,
                    ConstraintMatrix &constraints_dot) {
            auto &dof = this->get_dof_handler();
            auto &fe = this->get_fe();

            FEValuesExtractors::Vector displacements(0);
            FEValuesExtractors::Vector velocities(dim);
            ComponentMask displacement_mask = fe.component_mask(displacements);
            ComponentMask velocity_mask = fe.component_mask(velocities);

            // in 2D:
            // displacement_mask = [1 1 0 0 0]
            // velocity_mask     = [0 0 1 1 0]

            double timestep = this->get_current_time();
            double dt = this->get_timestep();

            // if timestep == nan workaround
            if (timestep != timestep) {
              timestep = 0;
            }
            if (dt != dt) {
              dt = 1;
            }
            // set d_dot to zero when the reference geometry is
            // transformed to the heart geometry in the first step.
            // this is the smarter version of #1 but it doesn't work 
            // yet.
            if (timestep == dt) {
              auto &solution_dot = const_cast<typename LAC::VectorType &>(
                  this->get_solution_dot());
              solution_dot.block(0) = 0;
            }

            // dirichlet BC for d
            if (dim == 2) {
              // bottom face
              heart_boundary_values(2, timestep);
              VectorTools::interpolate_boundary_values(
                  dof, 2, heart_boundary_values, *constraints[0],
                  displacement_mask);
              // left hull
              heart_boundary_values(0, timestep);
              VectorTools::interpolate_boundary_values(
                  dof, 0, heart_boundary_values, *constraints[0],
                  displacement_mask);
              // right hull
              heart_boundary_values(1, timestep);
              VectorTools::interpolate_boundary_values(
                  dof, 1, heart_boundary_values, *constraints[0],
                  displacement_mask);
              // top face
              heart_boundary_values(3, timestep);
              VectorTools::interpolate_boundary_values(
                  dof, 3, heart_boundary_values, *constraints[0],
                  displacement_mask);
            } else {
              // bottom face
              heart_boundary_values(1, timestep);
              VectorTools::interpolate_boundary_values(
                  dof, 1, heart_boundary_values, *constraints[0],
                  displacement_mask);
              // hull
              heart_boundary_values(0, timestep);
              VectorTools::interpolate_boundary_values(
                  dof, 0, heart_boundary_values, *constraints[0],
                  displacement_mask);
              // top face
              heart_boundary_values(2),
                  VectorTools::interpolate_boundary_values(
                      dof, 2, heart_boundary_values, *constraints[0],
                      displacement_mask);
            }
            int n_faces = (dim == 2) ? 3 : 2;
            // #1: set velocities and the time derivatice of the deformation
            // to zero, because the first step from the cylinder to the heart
            // shape is non physical
            if (timestep < 0.000001)
            {
              // time derivatives of dirichlet BC for d (called d_dot)
              for (int i = 0; i < n_faces; ++i) {
                VectorTools::interpolate_boundary_values(
                    dof, i, ZeroFunction<dim>(2 * dim + 1), constraints_dot,
                    displacement_mask);
              }
              // dirichlet BC for u
              for (int i = 0; i < n_faces; ++i) {
                VectorTools::interpolate_boundary_values(
                    dof, i, ZeroFunction<dim>(2 * dim + 1), *constraints[0],
                    velocity_mask);
              }
            } else {
              // time derivatives of dirichlet BC for d (called d_dot)
              for (int j = 0; j < n_faces; ++j) {
                heart_boundary_values(j, timestep, true);
                VectorTools::interpolate_boundary_values(
                    dof, j, heart_boundary_values, constraints_dot,
                    displacement_mask);
              }
              // dirichlet BC for u
              for (int j = 0; j < n_faces; ++j) {
                heart_boundary_values(j, timestep, true);
                VectorTools::interpolate_boundary_values(
                    dof, j, heart_boundary_values, *constraints[0],
                    velocity_mask);
              }
            }
          });
    } 
    // the "else" part is used when the heart geometry is turned off, in order to
    // run the test cases. Some adjustments had to be taken to get the error 
    // calculation right.
    else {
      // Make sure that velocity boundary conditions are applied on the Eulerian
      // domain.
      // This needs to be differnt w.r.t. the displacement variables, where
      // boundary conditions
      // are applied on the reference domain.
      signals.update_constraint_matrices.connect(
          [&, this](std::vector<std::shared_ptr<dealii::ConstraintMatrix>>
                        &constraints,
                    ConstraintMatrix &constraints_dot) {
            auto &dof = this->get_dof_handler();
            auto &fe = this->get_fe();

            FEValuesExtractors::Vector velocities(dim);
            ComponentMask velocity_mask = fe.component_mask(velocities);

            auto &dirichlet_bc = this->get_dirichlet_bcs();
            auto &dirichlet_bc_dot = this->get_dirichlet_bcs_dot();

            if (dirichlet_bc.get_mapped_ids().size() > 0) {
              AssertDimension(dirichlet_bc.get_mapped_ids().size(),
                              dirichlet_bc_dot.get_mapped_ids().size());

              auto boundary_id = dirichlet_bc.get_mapped_ids()[0];
              AssertDimension(1, dirichlet_bc.get_mapped_ids().size());
              AssertDimension(boundary_id,
                              dirichlet_bc_dot.get_mapped_ids()[0]);

              auto f = dirichlet_bc.get_mapped_function(boundary_id);
              auto f_dot = dirichlet_bc_dot.get_mapped_function(boundary_id);

              MappingQEulerian<dim, typename LAC::VectorType> mapping(
                  fe.degree, dof, this->get_locally_relevant_solution());

              VectorTools::interpolate_boundary_values(
                  mapping, dof, boundary_id, *f, *constraints[0],
                  velocity_mask);

              VectorTools::interpolate_boundary_values(
                  mapping, dof, boundary_id, *f_dot, constraints_dot,
                  velocity_mask);
            }
          });

      // Project or interpolate the initial conditions on the velocity using
      // a mapped geometry.
      signals.fix_initial_conditions.connect(
          [&, this](typename LAC::VectorType &y,
                    typename LAC::VectorType &y_dot) {
            auto &dof = this->get_dof_handler();
            auto &fe = this->get_fe();

            FEValuesExtractors::Vector velocities(dim);
            ComponentMask velocity_mask = fe.component_mask(velocities);

            auto &initial_solution = this->get_initial_solution();
            auto &initial_solution_dot = this->get_initial_solution_dot();

            MappingQEulerian<dim, typename LAC::VectorType> mapping(fe.degree,
                                                                    dof, this->get_locally_relevant_solution());

            if (fe.has_support_points()) {
              VectorTools::interpolate(mapping, dof, initial_solution, y,
                                       velocity_mask);
              VectorTools::interpolate(mapping, dof, initial_solution_dot,
                                       y_dot, velocity_mask);
            } else {
              AssertThrow(false, ExcNotImplemented());
            }
          });
    }
    // some arbitrary location to put the getter functions
    signals.begin_make_grid_fe.connect([&, this]() {
      adaptive_preconditioners_on = this->get_adaptive_preconditioners();
      max_iterations_adaptive = this->get_max_iterations_adaptive();
      use_explicit_solutions = this->get_explicit_solution_bool();
    });
    // retrieving information about the solver
    signals.end_solve_jacobian_system.connect([&, this]() {
      double time = this->get_current_time();
      if (time > 0.005 && std::is_same<LAC, LATrilinos>::value)
        iterations_last_step = this->get_solver_control()->last_step();
      else
        iterations_last_step = 0;
    });
    // destroy shared pointer...
    signals.end_run.connect([&, this]() {
      mapped_mapping = nullptr;
    });
  }
  // this function was once in the pi-DoMUS class..
  // it is used to map also the forcing terms on the Eulerian domain.
  virtual void apply_forcing_terms(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      FEValuesCache<dim, spacedim> &scratch,
      std::vector<double> &local_residual) const {

      auto mapping = MappingQEulerian<dim, typename LAC::VectorType, spacedim>(
              this->get_fe().degree, this->get_dof_handler(), this->get_locally_relevant_solution());

      const QGauss<dim> quadrature(this->get_fe().degree + 1);
      const QGauss<dim - 1> face_quadrature_formula(this->get_fe().degree + 1);
      FEValuesCache<dim, spacedim> mapped_scratch(
          mapping, this->get_fe(), quadrature, this->get_cell_update_flags(),
          face_quadrature_formula, this->get_face_update_flags());

    unsigned cell_id = cell->material_id();
    auto &forcing_terms = this->get_simulator().forcing_terms;
    if (forcing_terms.acts_on_id(cell_id)) {
      double dummy = 0.0;
      this->reinit(dummy, cell, scratch);
      this->reinit(dummy, cell, mapped_scratch);

      auto &fev = scratch.get_current_fe_values();
      auto &q_points = scratch.get_quadrature_points();
      auto &mapped_q_points = mapped_scratch.get_quadrature_points();

      auto &JxW = scratch.get_JxW_values();
      auto &mapped_JxW = mapped_scratch.get_JxW_values();
      auto qpsize = q_points.size();
      auto lrsize = local_residual.size();
      for (unsigned int q = 0; q < qpsize; ++q)
        for (unsigned int i = 0; i < lrsize; ++i) {
          for (unsigned int c = 0; c < dim; ++c) {
            double B = forcing_terms.get_mapped_function(cell_id)->value(
                q_points[q], c);
            local_residual[i] -=
                B * fev.shape_value_component(i, q, c) * JxW[q];
          }
          for (unsigned int c = dim; c < this->n_components; ++c) {
            double B = forcing_terms.get_mapped_function(cell_id)->value(
                mapped_q_points[q], c);
            local_residual[i] -=
                B * fev.shape_value_component(i, q, c) * mapped_JxW[q];
          }
        }
    }
  }


  virtual Mapping<dim,spacedim> &get_error_mapping() const {
    if(!mapped_mapping)
      mapped_mapping = SP(new MappingQEulerian<dim, typename LAC::VectorType, spacedim>
                             (this->get_fe().degree,
                              this->get_dof_handler(),
                              this->get_locally_relevant_solution()));
    return *(mapped_mapping.get());
  }

private:

  mutable std::shared_ptr<Mapping<dim,spacedim>> mapped_mapping;

  // Physical parameter
  double nu;  // this should actually be the dynamic viscosity...
  double rho;

  mutable unsigned int iterations_last_step;
  mutable unsigned int max_iterations_adaptive;
  mutable bool adaptive_preconditioners_on;
  mutable bool use_explicit_solutions;
  mutable bool disable_heart;

  bool Mp_use_inverse_operator;
  bool AMG_u_use_inverse_operator;
  bool AMG_d_use_inverse_operator;

  /**
   * AMG preconditioner for velocity-velocity matrix.
   */
  mutable ParsedAMGPreconditioner AMG_u;

  /**
   * AMG preconditioner for the pressure stifness matrix.
   */
  mutable ParsedAMGPreconditioner AMG_d;

  /**
   * Jacobi preconditioner for the pressure mass matrix.
   */
  mutable ParsedJacobiPreconditioner jac_M;

  /**
   * Heart boundary values. Data read only once to get as peed up. 
   */
  mutable BoundaryValues<dim> heart_boundary_values;
};

template <int dim, int spacedim, typename LAC>
ALENavierStokes<dim, spacedim, LAC>::ALENavierStokes()
    : PDESystemInterface<dim, spacedim, ALENavierStokes<dim, spacedim, LAC>,
                         LAC>("ALE Navier Stokes Interface", dim + dim + 1, 2,
                              "FESystem[FE_Q(2)^d-FE_Q(2)^d-FE_Q(1)]",
                              (dim == 2) ? "d,d,u,u,p" : "d,d,d,u,u,u,p",
                              "1,1,0"),
      AMG_u("AMG for u"), AMG_d("AMG for d"), jac_M("Jacobi for M") {
  this->init();
}

template <int dim, int spacedim, typename LAC>
void ALENavierStokes<dim, spacedim, LAC>::declare_parameters(
    ParameterHandler &prm) {
  PDESystemInterface<dim, spacedim, ALENavierStokes<dim, spacedim, LAC>,
                     LAC>::declare_parameters(prm);

  this->add_parameter(prm, &nu, "nu [Pa s]", "1.0", Patterns::Double(0.0),
                      "Viscosity");

  this->add_parameter(prm, &rho, "rho [kg m^-d]", "1.0", Patterns::Double(0.0),
                      "Density");

  this->add_parameter(prm, &Mp_use_inverse_operator,
                      "Invert Mp using inverse operator", "false",
                      Patterns::Bool(), "Invert Mp usign inverse operator");

  this->add_parameter(prm, &AMG_d_use_inverse_operator,
                      "AMG d - use inverse operator", "false", Patterns::Bool(),
                      "Enable the use of inverse operator for AMG d");

  this->add_parameter(prm, &AMG_u_use_inverse_operator,
                      "AMG u - use inverse operator", "false", Patterns::Bool(),
                      "Enable the use of inverse operator for AMG u");
}

template <int dim, int spacedim, typename LAC>
void ALENavierStokes<dim, spacedim, LAC>::parse_parameters_call_back() {}

template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void ALENavierStokes<dim, spacedim, LAC>::energies_and_residuals(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    FEValuesCache<dim, spacedim> &fe_cache, std::vector<EnergyType> &,
    std::vector<std::vector<ResidualType>> &residual,
    bool compute_only_system_terms) const {
  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Vector velocity(dim);
  const FEValuesExtractors::Scalar pressure(2 * dim);

  ResidualType et = 0;
  double dummy = 0.0;

  this->reinit(et, cell, fe_cache);

  /**
  * retrieving the different operators applied on functions from the fe_cache.
  */

  // displacement:
  auto &grad_ds =
      fe_cache.get_gradients("solution", "grad_d", displacement, et);
  auto &Fs =
      fe_cache.get_deformation_gradients("solution", "Fd", displacement, et);
  auto &ds_dot = fe_cache.get_values("solution_dot", "d_dot", displacement, et);
  auto &div_ds = fe_cache.get_divergences( "solution", "div_d", displacement, et);

  // explicit deformation gradients:
  auto &Fs_old = fe_cache.get_deformation_gradients("explicit_solution", "Fd",
                                                    displacement, dummy);

  // velocity:
  auto &us = fe_cache.get_values("solution", "u", velocity, et);
  auto &grad_us = fe_cache.get_gradients("solution", "grad_u", velocity, et);
  
  auto &sym_grad_us =
      fe_cache.get_symmetric_gradients("solution", "u", velocity, et);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", velocity, et);

  // Previous time step solution:
  auto &u_olds = fe_cache.get_values("explicit_solution", "u", velocity, dummy);

  // pressure:
  auto &ps = fe_cache.get_values("solution", "p", pressure, et);

  // Jacobian from the reference to the deformed cell:
  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_quad_points = us.size();

  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int quad = 0; quad < n_quad_points; ++quad) {
    // velocity:
    const Tensor<1, dim, ResidualType> &u_dot = us_dot[quad];
    const Tensor<2, dim, ResidualType> &grad_u = grad_us[quad];
    const Tensor<2, dim, ResidualType> &sym_grad_u = sym_grad_us[quad];

    // displacement
    const Tensor<1, dim, ResidualType> &d_dot = ds_dot[quad];
    const Tensor<2, dim, ResidualType> &grad_d = grad_ds[quad];
    const ResidualType &div_d = div_ds[quad];

    // deformation gradient, assigned differently due to different
    // ResidualTypes. Workaround to toggle the use of explicit and 
    // implicit solutions.
    Tensor<2, dim, ResidualType> F;
    if (use_explicit_solutions == true) {
      for (int d = 0; d < dim; ++d)
        for (int e = 0; e < dim; ++e)
          F[d][e] = Fs_old[quad][d][e];
    } else {
      F = Fs[quad];
    }

    ResidualType J = determinant(F);
    const Tensor<2, dim, ResidualType> &F_inv = invert(F);
    const Tensor<2, dim, ResidualType> &Ft_inv = transpose(F_inv);

    // Previous time step solution:
    const Tensor<1, dim, ResidualType> &u_old = u_olds[quad];

    // pressure:
    const ResidualType &p = ps[quad];

    // Jacobian of ALE transformation
    auto J_ale = J;

    // pressure * identity matrix
    Tensor<2, dim, ResidualType> p_Id;
    for (unsigned int i = 0; i < dim; ++i)
      p_Id[i][i] = p;
    // fluid stress tensor
    const Tensor<2, dim, ResidualType> sigma =
        -p_Id + nu * (sym_grad_u * F_inv + (Ft_inv * transpose(sym_grad_u)));

    for (unsigned int i = 0; i < residual[0].size(); ++i) {
      // test functions:
      // velocity:
      auto u_test = fev[velocity].value(i, quad);
      auto grad_u_test = fev[velocity].gradient(i, quad);

      // displacement:
      auto grad_d_test = fev[displacement].gradient(i, quad);
      auto div_d_test = fev[displacement].divergence(i, quad);

      // pressure:
      auto p_test = fev[pressure].value(i, quad);

      residual[1][i] += ((1. / nu) * p * p_test) * JxW[quad];
      // ALE Navier Stokes weak formulation as found almost exactly in 
      // Wick T. - Solvong Monolithic Fluid-Structure Interaction Problems in 
      // Arbitrary Lagrangian Eulerian Coordinates with the deal.II Library
      residual[0][i] +=
          (
              // time derivative term
              rho * scalar_product(u_dot * J_ale, u_test)
              // convection
              + rho * scalar_product(grad_u * (F_inv * (u_old - d_dot)) * J_ale, u_test)
              // diffusion
              + scalar_product(J_ale * sigma * Ft_inv, grad_u_test)
              // divergence free constriant
              - trace(grad_u * F_inv) * J_ale * p_test
              // solve linear elasticity problem
              + 1.0 * scalar_product(grad_d, grad_d_test)
              // mesh correction term, put to zero for better test results 
              // but not for the heart
              + 10 * (div_d * div_d_test)
          ) * JxW[quad];
    }
  }

  (void)compute_only_system_terms;
}

template <int dim, int spacedim, typename LAC>
void ALENavierStokes<dim, spacedim, LAC>::compute_system_operators(
    const std::vector<shared_ptr<LATrilinos::BlockMatrix>> matrices,
    LinearOperator<LATrilinos::VectorType> &system_op,
    LinearOperator<LATrilinos::VectorType> &prec_op,
    LinearOperator<LATrilinos::VectorType> &) const {
  typedef LATrilinos::VectorType::BlockType BVEC;
  typedef LATrilinos::VectorType VEC;

  // Preconditioners:
  const DoFHandler<dim, spacedim> &dh = this->get_dof_handler();
  const ParsedFiniteElement<dim, spacedim> fe = this->pfe;

  static int counter = 0;

  if (adaptive_preconditioners_on == true) {
    double time = this->get_current_time();
    // only refine at the beginning and if convergence is slow
    if (time <= 0.006 || iterations_last_step > max_iterations_adaptive) {
      AMG_d.initialize_preconditioner<dim, spacedim>(matrices[0]->block(0, 0),
                                                     fe, dh);
      AMG_u.initialize_preconditioner<dim, spacedim>(matrices[0]->block(1, 1),
                                                     fe, dh);
      jac_M.initialize_preconditioner<>(matrices[1]->block(2, 2));
      counter++;
    }
  } else {
    AMG_d.initialize_preconditioner<dim, spacedim>(matrices[0]->block(0, 0), fe,
                                                   dh);
    AMG_u.initialize_preconditioner<dim, spacedim>(matrices[0]->block(1, 1), fe,
                                                   dh);
    jac_M.initialize_preconditioner<>(matrices[1]->block(2, 2));
  }
  ////////////////////////////////////////////////////////////////////////////
  // SYSTEM MATRIX:

  std::array<std::array<LinearOperator<BVEC>, 3>, 3> S;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      S[i][j] = linear_operator<BVEC>(matrices[0]->block(i, j));
  system_op = BlockLinearOperator<VEC>(S);

  ////////////////////////////////////////////////////////////////////////////
  // PRECONDITIONER MATRIX:

  std::array<std::array<LinearOperator<TrilinosWrappers::MPI::Vector>, 3>, 3> P;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      P[i][j] = linear_operator<BVEC>(matrices[0]->block(i, j));

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<BVEC> solver_CG(solver_control_pre);
  static SolverGMRES<BVEC> solver_GMRES(solver_control_pre);

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      if (i != j)
        P[i][j] = null_operator<TrilinosWrappers::MPI::Vector>(P[i][j]);

  auto A = linear_operator<BVEC>(matrices[0]->block(1, 1));
  auto B = linear_operator<BVEC>(matrices[0]->block(2, 1));
  auto Bt = transpose_operator<>(B);

  LinearOperator<BVEC> A_inv;
  if (AMG_u_use_inverse_operator) {
    A_inv = inverse_operator(S[1][1], solver_GMRES, AMG_u);
  } else {
    A_inv = linear_operator<BVEC>(matrices[0]->block(1, 1), AMG_u);
  }

  auto Mp =
      linear_operator<TrilinosWrappers::MPI::Vector>(matrices[1]->block(2, 2));

  LinearOperator<BVEC> Mp_inv;
  if (Mp_use_inverse_operator) {
    Mp_inv = inverse_operator(Mp, solver_GMRES, jac_M);
  } else {
    Mp_inv = linear_operator<BVEC>(matrices[1]->block(2, 2), jac_M);
  }

  auto Schur_inv = nu * Mp_inv;

  if (AMG_d_use_inverse_operator) {
    P[0][0] = inverse_operator(S[0][0], solver_CG, AMG_d);
  } else {
    P[0][0] = linear_operator<BVEC>(matrices[0]->block(0, 0), AMG_d);
  }

  P[1][1] = A_inv;
  P[1][2] = A_inv * Bt * Schur_inv;
  P[2][1] = null_operator(B);
  P[2][2] = -1 * Schur_inv;

  prec_op = BlockLinearOperator<VEC>(P);
}

#endif

/*! @} */
