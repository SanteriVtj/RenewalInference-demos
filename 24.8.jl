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

# ╔═╡ 196322d5-63f5-40b7-9bce-d72da3915eda
using GLMakie, CairoMakie, PlutoUI, RenewalInference, QuasiMonteCarlo, ForwardDiff, Distributions, HypertextLiteral, LaTeXStrings

# ╔═╡ 428842e3-cc58-4b4e-9e2b-343fbba6a02e
# Unpack parameters for normal
function parametrisation_wrapper(par, T=1, N=100)
	ϕ, γ, σᵢ = par
	par = [ϕ, γ, σᵢ, 0, 0]
	x = RenewalInference.log_norm_parametrisation(par,T)
	return [x[1][1], x[2][1]]
end

# ╔═╡ eb415db6-cc5a-487c-810f-fe3f7be9fb4d
begin
	md"""
	ϕ: $(@bind ϕ₁ PlutoUI.Slider(0:.01:1, show_value = true, default = .75))
	
	γ: $(@bind γ₁ PlutoUI.Slider(0:.01:1, show_value = true, default = .1))

	σᵢ: $(@bind σ₁ PlutoUI.Slider(1_000:100_000, show_value = true, default = 20_000.))
	"""
	# β₁: $(@bind innovator_β₁1 PlutoUI.Slider(0:100, show_value = true, default = 15.))
	
	# β₂: $(@bind innovator_β₂1 PlutoUI.Slider(0:100, show_value = true, default = 5.))
end

# ╔═╡ 52703390-0d26-45c2-ba2d-0177239c0ee4
begin
	# Compute Jacobian w.r.t. ϕ, σⁱ and γ
	J = ForwardDiff.jacobian(a->parametrisation_wrapper(a), [ϕ₁, γ₁, σ₁])

	# Derivatives w.r.t μ & σ
	J₁, J₂ = J[1,:], J[2,:];
end

# ╔═╡ d068cbc2-9679-48c1-88dc-b574cb65f02d
begin
	labels = [L"\phi", L"\gamma", L"\sigma_i"]
	
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

# ╔═╡ Cell order:
# ╟─35612b5e-3c43-11ee-0f02-21dc93992596
# ╟─b5124229-065c-459e-b5d5-c14cb3eabeca
# ╟─196322d5-63f5-40b7-9bce-d72da3915eda
# ╠═428842e3-cc58-4b4e-9e2b-343fbba6a02e
# ╟─eb415db6-cc5a-487c-810f-fe3f7be9fb4d
# ╟─52703390-0d26-45c2-ba2d-0177239c0ee4
# ╟─d068cbc2-9679-48c1-88dc-b574cb65f02d
