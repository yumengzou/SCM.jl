using DataFrames, CSV
using Statistics
using LinearAlgebra
using JuMP, Ipopt
using Optim

function loss_V(v; Z1_scaled, Z0_scaled, Y1, Y0)
    if abs(sum(v)) < 1e-6
        return 1e6
    end
    num_controls = size(Z0_scaled)[2];
    num_prior_periods = size(Y1)[1];
    
    # make sure vector v is a weight vector
    # create a diagonal matrix out of the vector v
    V = Diagonal(( abs.(v)./sum(abs.(v)) ));
    
    # Solve the w that minimizes (Z₁-Z₀*w)'*V*(Z₁-Z₀*w) for a given V
    model = Model(Ipopt.Optimizer);
    set_silent(model);
    H = Z0_scaled' * V * Z0_scaled;
    c = -1 * vec(Z1_scaled' * V * Z0_scaled);
    @variable(model, 0 <= w[1:num_controls] <= 1);
    @constraint(model, sum(w) == 1);
    @objective(model, Min, c' * w + 0.5 * w' * H * w);
    optimize!(model);
    w_sol= value.(w);

    # compute the Mean Squared Prediction Error
    v_loss = (Y1 - Y0 * w_sol)' * (Y1 - Y0 * w_sol);
    v_loss = only(v_loss) / num_prior_periods;
    return v_loss
end

function synth(Z1, Z0, Y1, Y0)
    Z1 = Matrix(Z1);
    Z0 = Matrix(Z0);
    Y1 = Matrix(Y1);
    Y0 = Matrix(Y0);
    
    num_predictors = size(Z1)[1];
    num_controls = size(Z0)[2];
    num_prior_periods = size(Y1)[1];
    
    # normalize predictors
    Z = hcat(Z1, Z0);
    predictor_sd = vec(sqrt.(var(Matrix(Z), dims=2)));
    Z_scaled = transpose(Z' * Diagonal((1 ./ predictor_sd)));
    Z1_scaled = Z_scaled[:, 1];
    Z0_scaled = Z_scaled[:, 2:(num_controls+1)];

    # optimize V, using different starting values
    ## equal weight starting value
    v_start1 = fill(1/num_predictors, num_predictors);
    l(v) = loss_V(v; Z1_scaled = Z1_scaled, Z0_scaled = Z0_scaled, Y1 = Y1, Y0 = Y0);
    v_opt1 = optimize(
        l, v_start1, 
        Optim.Options(iterations = 2000)
    );

    ## regression-based starting values
    Z = hcat(Z1_scaled, Z0_scaled);
    Z = hcat(fill(1, num_controls+1), Z');
    Y = hcat(Y1, Y0)';
    betas = inv(Z' * Z) * (Z' * Y);      # βs for all pre-treatment period outcome
    betas = betas[2:(num_predictors+1), :]; # remove the β₀ for the constant term
    beta_sq_sum = diag(betas * betas');        # square and then sum βₖs across the pre-treatment period 
    v_start2 = beta_sq_sum / sum(beta_sq_sum);
    v_opt2 = optimize(
        l, v_start2, 
        Optim.Options(iterations = 2000)
    );
    
    # pick the best V
    v_loss1 = Optim.minimum(v_opt1);
    v_loss2 = Optim.minimum(v_opt2);
    if v_loss1 < v_loss2
        v_sol = Optim.minimizer(v_opt1);
        v_loss = v_loss1
    else 
        v_sol = Optim.minimizer(v_opt2);
        v_loss = v_loss2
    end
    v_sol = abs.(v_sol)./sum(abs.(v_sol));

    # optimize W given V
    model = Model(Ipopt.Optimizer);
    set_silent(model);
    V = Diagonal(v_sol);
    H = Z0_scaled' * V * Z0_scaled;
    c = -1 * vec(Z1_scaled' * V * Z0_scaled);
    @variable(model, 0 <= w[1:num_controls] <= 1);
    @constraint(model, sum(w) == 1);
    @objective(model, Min, c' * w + 0.5 * w' * H * w);
    optimize!(model);
    w_sol= value.(w);
    w_loss = objective_value(model);
    ## add the constant term to make it a norm 0.5(Z₁-Z₀*W)'V(Z₁-Z₀*W) that's comparable to zero
    w_loss = w_loss + 0.5 * Z1_scaled' * V * Z1_scaled 
    return Dict(
        "v" => v_sol,
        "w" => w_sol,
        "v_loss" => v_loss,
        "w_loss" => w_loss,
    )
end