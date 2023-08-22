### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 35612b5e-3c43-11ee-0f02-21dc93992596
using Pkg

# ╔═╡ b5124229-065c-459e-b5d5-c14cb3eabeca
Pkg.activate("C:/Users/Santeri/Desktop/RenewalInference-demos")

# ╔═╡ 69c14d12-73a1-4f52-ac81-1d26c4cba3a2
Pkg.develop(path="C:/Users/Santeri/.julia/dev/RenewalInference.jl")

# ╔═╡ 196322d5-63f5-40b7-9bce-d72da3915eda
using GLMakie, CairoMakie, PlutoUI, RenewalInference, QuasiMonteCarlo, ForwardDiff, Distributions, HypertextLiteral, LaTeXStrings

# ╔═╡ 428842e3-cc58-4b4e-9e2b-343fbba6a02e
# Unpack parameters for normal
function parametrisation_wrapper(par, T=1)
	ϕ, σᵢ, γ = par
	par = [ϕ, σᵢ, γ, 0, 0]
	x = RenewalInference.log_norm_parametrisation(par,T)
	return [x[1][1], x[2][1]]
end

# ╔═╡ eb415db6-cc5a-487c-810f-fe3f7be9fb4d
begin
	md"""
	ϕ: $(@bind ϕ₁ PlutoUI.Slider(0:.01:1, show_value = true, default = .75))
	
	σᵢ: $(@bind σ₁ PlutoUI.Slider(1_000:100_000, show_value = true, default = 20_000.))
	
	γ: $(@bind γ₁ PlutoUI.Slider(0:.01:1, show_value = true, default = .1))
	"""
	# β₁: $(@bind innovator_β₁1 PlutoUI.Slider(0:100, show_value = true, default = 15.))
	
	# β₂: $(@bind innovator_β₂1 PlutoUI.Slider(0:100, show_value = true, default = 5.))
end

# ╔═╡ 52703390-0d26-45c2-ba2d-0177239c0ee4
begin
	# Compute Jacobian w.r.t. ϕ, σⁱ and γ
	J = ForwardDiff.jacobian(a->parametrisation_wrapper(a), [ϕ₁, σ₁, γ₁])

	# Derivatives w.r.t μ & σ
	J₁, J₂ = J[1,:], J[2,:];
end

# ╔═╡ d068cbc2-9679-48c1-88dc-b574cb65f02d
begin
	labels = [L"\phi", L"\sigma_i", L"\gamma"]
	
	f = Figure()
	μ_ax = Axis(f[1,1], 
		xlabel = "w.r.t. variable", 
		ylabel = L"\partial_x\mu",
		xticks = (1:3,labels)
	)
	σ_ax = Axis(f[2,1], 
		xlabel = "w.r.t. variable", 
		ylabel = L"\partial_x\sigma",
		xticks = (1:3,labels)
	)

	barplot!(
		μ_ax,
		1:3,
		J₁
	)
	
	barplot!(
		σ_ax,
		1:3,
		J₂
	)
	f
end

# ╔═╡ 04dc55e7-5a1d-4ba3-b009-a6852bfe49c0
begin
	md"""
	δ: $(@bind δ₁ PlutoUI.Slider(0:.01:1, show_value = true, default = .75))
	
	θ: $(@bind θ₁ PlutoUI.Slider(0:.01:1, show_value = true, default = .1))

	first cost: $(@bind c PlutoUI.Slider(0:1000, show_value = true, default = .1))
	"""
end

# ╔═╡ 170e925c-d87c-47a0-a572-279b5c715176
function threshold_wrapper(par, c, N=200, T=5, β=.95, alg=QuasiMonteCarlo.HaltonSample())
	δ, θ = par
	par = [0,0,0,δ,θ]
	obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = quantile.(Normal(), QuasiMonteCarlo.sample(N,1,alg)')
	return RenewalInference.thresholds(par, collect(c), ishock, obsolence, β)
end

# ╔═╡ 56cd3294-aa61-433e-860a-cb3a1221edb0
J2 = ForwardDiff.jacobian(a->threshold_wrapper(a,c:10:(c+40)), [δ₁, θ₁])

# ╔═╡ 75c00986-dca4-48d3-975b-8856c8570d40
threshold_wrapper([δ₁,θ₁], c:10:(c+40))

# ╔═╡ edf60fd3-788b-4f9e-98eb-10f7e82e93a6
begin
	md"""
	ϕ: $(@bind ϕ₂ PlutoUI.Slider(0:.01:1, show_value = true, default = .75))
	
	σᵢ: $(@bind σ₂ PlutoUI.Slider(1_000:100_000, show_value = true, default = 20_000.))
	
	γ: $(@bind γ₂ PlutoUI.Slider(0:.01:1, show_value = true, default = .1))
	
	δ: $(@bind δ₂ PlutoUI.Slider(0:.01:1, show_value = true, default = .95))
	
	θ: $(@bind θ₂ PlutoUI.Slider(0:.01:1, show_value = true, default = .95))

	first cost: $(@bind c2 PlutoUI.Slider(0:1000, show_value = true, default = .1))
	"""
end

# ╔═╡ 74f3de4d-5169-4310-b09f-2ef2785f5f9c
function patent_wrapper(par, 
	c, β=.95, ν=2, N=200, T=17, alg=QuasiMonteCarlo.HaltonSample())
	sim_param = [i.value for i in deepcopy(par)]
	simulation_shocks = QuasiMonteCarlo.sample(N,T,alg)'
	obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = quantile.(Normal(), QuasiMonteCarlo.sample(N,1,alg)')
	cost = collect(c)
	hz=RenewalInference.simulate_patenthz(sim_param,simulation_shocks, obsolence,cost,ishock)[1]
	return RenewalInference.patenthz(
        par,
        hz,
        ishock,
        obsolence,
        cost,
        β=β,
        ν=ν
	)[1]
end

# ╔═╡ 41f6f212-2242-4a2e-a2dc-ec66e2c03e78
begin
	costs = [116, 138, 169, 201, 244, 286, 328, 381, 445, 508, 572, 646, 720, 794, 868, 932, 995];
	J3 = ForwardDiff.gradient(a->patent_wrapper(a,costs), [ϕ₂, σ₂, γ₂, δ₂, θ₂])

	labels2 = [L"\phi", L"\sigma_i", L"\gamma", L"\delta", L"\theta"]
	fig = Figure()
	ax = Axis(fig[1,1], 
		xlabel = "w.r.t. variable", 
		ylabel = L"\partial_x\mathcal{F}",
		xticks = (1:5,labels2)
	)

	barplot!(
		ax,
		1:5,
		J3
	)
	fig
end

# ╔═╡ Cell order:
# ╟─35612b5e-3c43-11ee-0f02-21dc93992596
# ╟─b5124229-065c-459e-b5d5-c14cb3eabeca
# ╟─69c14d12-73a1-4f52-ac81-1d26c4cba3a2
# ╟─196322d5-63f5-40b7-9bce-d72da3915eda
# ╟─428842e3-cc58-4b4e-9e2b-343fbba6a02e
# ╟─eb415db6-cc5a-487c-810f-fe3f7be9fb4d
# ╠═52703390-0d26-45c2-ba2d-0177239c0ee4
# ╟─d068cbc2-9679-48c1-88dc-b574cb65f02d
# ╟─04dc55e7-5a1d-4ba3-b009-a6852bfe49c0
# ╠═170e925c-d87c-47a0-a572-279b5c715176
# ╠═56cd3294-aa61-433e-860a-cb3a1221edb0
# ╠═75c00986-dca4-48d3-975b-8856c8570d40
# ╟─edf60fd3-788b-4f9e-98eb-10f7e82e93a6
# ╟─74f3de4d-5169-4310-b09f-2ef2785f5f9c
# ╟─41f6f212-2242-4a2e-a2dc-ec66e2c03e78
