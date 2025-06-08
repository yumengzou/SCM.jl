using DataFrames, CSV
using Statistics
using LinearAlgebra
using JuMP, Ipopt
using Optim

function loss_V(v; Z₁, Z₀, Y₁, Y₀)
    if abs(sum(v)) < 1e-6
        return 1e6
    end
    num_controls = size(Z₀)[2];
    T₀ = size(Y₁)[1];
    
    # make sure vector v is a weight vector
    # create a diagonal matrix out of the vector v
    V = Diagonal(( abs.(v)./sum(abs.(v)) ));
    
    # Solve the w that minimizes (Z₁-Z₀*w)'*V*(Z₁-Z₀*w) for a given V
    model = Model(Ipopt.Optimizer);
    set_silent(model);
    H = Z₀' * V * Z₀;
    c = -1 * vec(Z₁' * V * Z₀);
    @variable(model, 0 <= w[1:num_controls] <= 1);
    @constraint(model, sum(w) == 1);
    @objective(model, Min, c' * w + 0.5 * w' * H * w);
    optimize!(model);
    w_sol= value.(w);

    # compute the Mean Squared Prediction Error
    v_loss = (Y₁ - Y₀ * w_sol)' * (Y₁ - Y₀ * w_sol);
    v_loss = only(v_loss) / T₀;
    return v_loss
end

function synth(Z1, Z0, Y1, Y0)
    Z1 = Matrix(Z1);
    Z0 = Matrix(Z0);
    Y1 = Matrix(Y1);
    Y0 = Matrix(Y0);
    
    num_predictors = size(Z1)[1];
    num_controls = size(Z0)[2];
    T₀ = size(Y1)[1];
    
    # normalize predictors
    Z = hcat(Z1, Z0);
    predictor_sd = vec(sqrt.(var(Matrix(Z), dims=2)));
    Z_scaled = transpose(Z' * Diagonal((1 ./ predictor_sd)));
    Z1_scaled = Z_scaled[:, 1];
    Z0_scaled = Z_scaled[:, 2:(num_controls+1)];

    # optimize V, using different starting values
    ## equal weight starting value
    v_start1 = fill(1/num_predictors, num_predictors);
    l(v) = loss_V(v; Z₁ = Z1_scaled, Z₀ = Z0_scaled, Y₁ = Y1, Y₀ = Y0);
    lower = fill(-10, num_predictors);
    upper = fill(10, num_predictors);
    v_opt1 = optimize(
        l, lower, upper, v_start1, Fminbox(LBFGS()),
        Optim.Options(x_abstol = 5e-4, f_abstol = 1e-5, iterations = 1000)
    );

    ## regression-based starting values
    Z = hcat(Z1_scaled, Z0_scaled);
    Z = hcat(fill(1, num_controls+1), Z');
    Y = hcat(Y1, Y0)';
    βs = inv(Z' * Z) * (Z' * Y);      # βs for all pre-treatment period outcome
    βs = βs[2:(num_predictors+1), :]; # remove the β₀ for the constant term
    β_sq_sum = diag(βs * βs');        # square and then sum βₖs across the pre-treatment period 
    v_start2 = β_sq_sum / sum(β_sq_sum);
    v_opt2 = optimize(
        l, lower, upper, v_start2, Fminbox(LBFGS()),
        Optim.Options(x_abstol = 5e-4, f_abstol = 1e-5, iterations = 1000)
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