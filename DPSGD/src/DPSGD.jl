module DPSGD

export DifferentialPrivacy, privacy_spent
export solve_niterations, solve_noise_multiplier

using
    Flux,
    LogarithmicNumbers,
    Parameters,
    Random,
    Statistics,
    Zygote,
    LinearAlgebra

using SpecialFunctions: logerfc, gamma

Base.binomial(n, k) = gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

"""
    DifferentialPrivacy(nexamples; ...)

Clip gradients when their absolute value exceeds `clip_threshold`.
"""
@with_kw mutable struct DifferentialPrivacy
    clip_threshold::Float32 = 1.0
    noise_multiplier::Float32 = 1.0
    lotsize::Int = 0
    nexamples::Int
    niterations::Int = 0
    first_parameters = nothing
    norm::Vector{<:AbstractFloat} = zeros(1)
end

function DifferentialPrivacy(nexamples; lotsize = 0, kw...)
    DifferentialPrivacy(; nexamples = nexamples, lotsize = lotsize, kw...)
end

function calculate_norms!(opt, xs::Params, gs)
    per_parameter_norms = nothing
    num_samples = 0
    grad_float_type = AbstractFloat # Specific type will be inferred from gs below
    for x in xs
        isnothing(gs[x]) && continue
        if num_samples == 0
            num_samples = size(gs[x])[1]
            grad_float_type = eltype(gs[x])
        else
            @assert num_samples == size(gs[x])[1]
        end
        parameter_norm_per_sample = [norm(gs[x][i,:]) for i in 1:num_samples]
        if isnothing(per_parameter_norms)
            per_parameter_norms = parameter_norm_per_sample
        else
            per_parameter_norms = hcat(per_parameter_norms, parameter_norm_per_sample)
        end
       
    end

    # Norm Across Params
    per_sample_norm = zeros(grad_float_type, num_samples)
    for i in 1:num_samples
        per_sample_norm[i] = norm(per_parameter_norms[i,:])
    end

    # Store in DifferentialPrivacy Struct
    for sub_opt in opt.os
        if isa(sub_opt, DifferentialPrivacy)
            sub_opt.norm = per_sample_norm
        end
    end
end

"""
    train!(loss, ps, data, opt::DifferentialPrivacy; ...)

Calculate the per example gradient when the loss
is a vector with an element per example.
"""
function train!(loss, ps, data, opt; cb = () -> ())
    ps = Zygote.Params(ps)
    cb = Flux.Optimise.runall(cb)

    for d in data
        try
            gs = Zygote.jacobian(ps) do
                loss(Flux.Optimise.batchmemaybe(d)...)
            end
            calculate_norms!(opt, ps, gs)
            Flux.update!(opt, ps, gs)
            cb()
        catch ex
            if ex isa Flux.Optimise.StopException
                break
            elseif ex isa Flux.Optimise.SkipException
                continue
            else
                rethrow(ex)
            end
        end
    end
end

Flux.train!(l, p, d, opt::DifferentialPrivacy; cb = () -> ()) = train!(l, p, d, opt; cb)

# This is definitely type piracy...but not in *spirit*, right? It's like, Samuel
# Bellamy-style type piracy. Flux.Optimiser is the thieving rich, and we're just
# stealing DifferentialPrivacy back.
function Flux.train!(l, p, d, opt::Flux.Optimiser; cb = () -> ())
    f = any(x -> isa(x, DifferentialPrivacy), opt.os) ? train! : Flux.train!
    f(l, p, d, opt; cb)
end

function Flux.Optimise.apply!(opt::DifferentialPrivacy, x, Δ)
    if opt.lotsize ≠ size(Δ, 1)
        if opt.lotsize == 0
            opt.lotsize = size(Δ, 1)
            @debug "Setting lot size to $(opt.lotsize)"
        else
            @warn "Lot size appears to be dynamic, make sure to specify it in the constructor or results will be invalid." maxlog = 1
        end
    end

    # Gradient clipping
    for i in 1:length(opt.norm)
        # Δ_norm = Flux.norm(Δ)
        Δ_norm = opt.norm[i]
        if Δ_norm > opt.clip_threshold
            Δ[i,:] = Flux.rmul!(Δ[i,:], opt.clip_threshold / Δ_norm)
        end
    end
    # Accumulate gradients across examples
    Δ = reshape(Statistics.mean(Δ; dims = 1), size(x))

    # Add noise
    a = similar(Δ)
    Random.randn!(a)
    a *= (opt.noise_multiplier * opt.clip_threshold)
    Δ += a

    # Only increment iterations once per model iteration, not for every
    # parameter set
    if isnothing(opt.first_parameters)
        opt.first_parameters = x
    end

    if opt.first_parameters === x
        opt.niterations += 1
    end

    return Δ
end

function privacy_spent(dp::DifferentialPrivacy, target_delta; kwargs...)
    privacy_spent(target_delta;
        lotsize = dp.lotsize,
        nexamples = dp.nexamples,
        noise_multiplier = dp.noise_multiplier,
        niterations = dp.niterations,
        kwargs...)
end

function privacy_spent(target_delta;
    nexamples, noise_multiplier, niterations,
    lotsize = 1,
    orders = 1:64)
    q = lotsize / nexamples
    rdp = renyi_differential_privacy.((q,), (noise_multiplier,), orders; niterations)

    if length(orders) ≠ length(rdp)
        throw(ArgumentError("Input lists must have the same length."))
    end

    eps = (rdp
           .-
           ((log.(target_delta) .+ log.(orders)) ./ (orders .- 1))
           .+
           (log.((orders .- 1) ./ orders)))

    # Special case when there is no privacy.
    all(isnan, eps) && return Inf

    replace!(eps, NaN => Inf)
    return minimum(eps)
end

function renyi_differential_privacy(q, σ, α; niterations = 1)
    iszero(q) && return 0.0
    iszero(σ) && return Inf # No privacy.
    α == Inf && return Inf
    isone(q) && return α / (2 * σ^2)

    return log(compute_A(q, σ, α)) / (α - 1) * niterations
end

# Refers to $A_α$ in RDP.
function compute_A(q, σ, α)
    if isinteger(α)
        compute_A(Integer, q, σ, Integer(α))
    else
        compute_A(Real, q, σ, α)
    end
end

# Notes on `Logarithmic`:
# - `Logarithmic(x)` is the number `x`, but represented inside the structure
#    by `log(x)` for the sake of accurate arithmetic when `x` is large.
# - Calling `log` on a `Logarithmic{T}` number simply unwraps the value already
#   being used to represent the number (it returns a value of type `T`).
# - If you know `a = log(x)`, then `Logarithmic(x)` can be constructed
#   with no calculation using `exp(Logarithmic, a)`.
# - Negative values *can* be represented with `Logarithmic` without doing
#   anything special--it handles managing the sign for you.

# https://arxiv.org/pdf/1908.10530.pdf
# Section 3.3, Case I: Integer α
function compute_A(::Type{Integer}, q, σ, α::Integer)
    q = Logarithmic(q)
    sum(0:α; init = Logarithmic(0)) do i
        c_i = binomial(α, i) * q^i * (1 - q)^(α - i)
        c_i * exp(Logarithmic, (i^2 - i) / (2σ^2))
    end
end

# Section 3.3, Case II: Fractional
function compute_A(::Type{Real}, q, σ, α; tol = 1e-14)
    q = Logarithmic(q)
    # The two parts of A_alpha are integrals over (-∞, z] and [z, ∞).
    z = 1 / 2 + ((1 - q) / q)σ^2

    # For integers, `binomial(n, k)` for `k>n` is zero, but that is not true for
    # non-integer values of `k`. So this is an infinite sum, carried out to the
    # desired precision.
    i = 0
    total = Logarithmic(0.0)
    last_term = tol + 1.0
    while last_term > tol
        b = binomial(α, i)

        c1 = b * q^i * (1 - q)^(α - i)
        c2 = b * q^(α - i) * (1 - q)^i

        e1 = 1 / 2 * exp(Logarithmic, logerfc((i - z) / (√2σ)))
        e2 = 1 / 2 * exp(Logarithmic, logerfc((z - (α - i)) / (√2σ)))

        s1 = c1 * e1 * exp(Logarithmic, (i^2 - i) / (2σ^2))
        s2 = c2 * e2 * exp(Logarithmic, ((α - i)^2 - (α - i)) / (2σ^2))

        last_term = s1 + s2
        total += last_term
        i += 1
    end
    return total
end

function solve_noise_multiplier(; epsilon, delta,
    nexamples, niterations,
    lotsize = 1, orders = 1:64, lo = 1e-15, hi = 1e16, tol = 1e-6,
    kwargs...)
    # Adding less noise to get the same privacy is better, so find the minimum σ
    # over all `orders`.
    minimum(orders) do order
        # The necessary noise for this `order` is somewhere in between these
        # bounds, so we're forced to keep σ_hi.
        σ_lo, σ_hi = binary_search(epsilon; lo = lo, hi = hi,
                                            tol = tol, lt = (>), kwargs...) do σ
            privacy_spent(delta;
                lotsize = lotsize,
                nexamples = nexamples,
                noise_multiplier = σ,
                orders = order:order,
                niterations = niterations)
        end
        σ_hi
    end
end

function solve_niterations(; epsilon, delta,
    nexamples, noise_multiplier,
    lotsize = 1, orders = 1:64, lo = 0, hi = 2^63 - 1, tol = 1,
    kwargs...)
    # See comments for `solve_noise_multiplier`. The logic is the same, except
    # reversed because being able to do more iterations is better.
    maximum(orders) do order
        n_lo, n_hi = binary_search(epsilon; lo = lo, hi = hi, tol = tol, kwargs...) do n
            privacy_spent(delta;
                lotsize = lotsize,
                nexamples = nexamples,
                noise_multiplier = noise_multiplier,
                orders = order:order,
                niterations = round(Int, n))
        end
        isfinite(n_lo) ? floor(Int, n_lo) : -Inf
    end
end

function binary_search(f, y;
    lo = -1e16,
    hi = 1e16,
    maxtries = 10000,
    tol = 1e-12,
    atol = 1e-12,
    lt = isless)
    for i in 1:maxtries
        (hi - lo ≤ tol) && break

        x = (lo + hi) / 2

        yhat = f(x)
        if isapprox(yhat, y; atol = atol)
            break
        end

        # Since the caller can override `lt`, assume a monotonically increasing
        # function.
        if lt(yhat, y)
            # f(x) is too small, so x is the new lower bound
            lo = x
        else
            hi = x
        end
    end

    return if lt(f(lo), y) && lt(y, f(hi))
        (lo, hi)
    elseif lt(f(hi), y)
        (hi, Inf)
    else
        (-Inf, lo)
    end
end

end # module
