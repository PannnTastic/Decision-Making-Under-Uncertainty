### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ bec4bdb1-b9cf-44df-8c20-74e75f8d0e5d
using POMDPs, QuickPOMDPs, POMDPModelTools, BeliefUpdaters, Parameters

# ╔═╡ e6ad5876-cf5e-453a-96b6-bd15996a7fdb
using POMDPPolicies

# ╔═╡ f83fa2ae-5532-45d5-86d2-e2bb26e47285
using QMDP

# ╔═╡ c542c586-bb71-4509-9c5c-5d5870cb53e3
using FIB

# ╔═╡ 6e1c8f95-4c62-4b97-b6b8-f76455a867e1
using PointBasedValueIteration

# ╔═╡ 8ed40b93-2bc4-4e2a-87c0-5936516813d4
# ╠═╡ show_logs = false
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

# ╔═╡ f8a3389c-3cec-4438-98cc-34b0a70025d3
using StatsBase

# ╔═╡ af668bdd-adab-48b0-ba41-c42b49d5c39b
using PlutoUI

# ╔═╡ 7bb6ea08-5ed2-4706-9b51-06b88106ea64
# ╠═╡ show_logs = false
using BasicPOMCP

# ╔═╡ 58a8e286-3537-4b10-96b5-7a294b2f3b08
using JSON

# ╔═╡ b2c89f40-d530-4dca-bb25-76f85028506e
using D3Trees

# ╔═╡ b3619d30-78a2-11f1-068c-7974b7ab15b3
md"# HandsOn POMDP"

# ╔═╡ d6a04964-52b3-4f5e-a00d-fd3153cfc3fd
md"""
A partially observable Markov decision process (POMDP) is a 7-tuple consisting of:

$\langle \mathcal{S}, \mathcal{A}, {\color{blue}\mathcal{O}}, T, R, {\color{blue}O}, \gamma \rangle$

Variable           | Description          | `POMDPs` Interface
:----------------- | :------------------- | :-------------------:
$\mathcal{S}$      | State space          | `POMDPs.states`
$\mathcal{A}$      | Action space         | `POMDPs.actions`
$\mathcal{O}$      | Observation space    | `POMDPs.observations`
$T$                | Transition function  | `POMDPs.transision`
$R$                | Reward function      | `POMDPs.reward`
$O$                | Observation function | `POMDPs.observation`
$\gamma \in [0,1]$ | Discount factor      | `POMDPs.discount`

Notice the addition of the observation space $\mathcal{O}$ and observation function $O$, which differ from the MDP 5-tuple. Rememeber, the agent receives _observations_ of the current state rather than the true state—and using past observations, it builds a _belief_ of the underlying state (this can be represented as a probability distribution over true states).
"""

# ╔═╡ f1d195d6-5776-4056-95b4-d4392f5ef9f9
md"""
## Crying Baby Problem (POMDP)
The *crying baby problem* is a simple POMDP with two states $\{\rm hungry, full\}$, two actions $\{\rm feed, ignore\}$, and two observations $\{\rm crying, quiet\}$.

$$\begin{align}
\mathcal{S} &= \{\rm hungry, full\}\\
\mathcal{A} &= \{\rm feed, ignore\}\\
\mathcal{O} &= \{\rm crying, quiet\}
\end{align}$$
"""

# ╔═╡ 5050d20c-d41c-440e-a78f-9b4d644434c4
md" ## Environment Setup"

# ╔═╡ 9e6f9788-f932-4f7a-9aa3-4c8def87e8b0
@with_kw struct CryingBabyParameters
 	#rewards
 	r_hungry = -10
 	r_feed 	 = -5
 	#transition probability
 	p_become_hungry::Real = 0.1
 	#observation probability 
 	p_crying_when_hungry::Real = 0.8
 	p_crying_when_full::Real = 0.1
end

# ╔═╡ fc2971ca-bfdd-40f9-ac50-22681036e520
params = CryingBabyParameters();

# ╔═╡ f49ad3a7-8da0-4723-9f81-23d6fe3f5c45
md"""
## State, Action, and Observation
Next we define our states,action, and observation"""

# ╔═╡ 5adef0b9-fe46-402e-bbbc-29a460b13831
begin
	@enum State HUNGRYₛ FULLₛ
	@enum Action FEEDₐ IGNOREₐ
	@enum Observation CRYINGₒ QUIETₒ
end

# ╔═╡ c7664ad5-0a94-4813-b8d1-008369b50d78
md"### State"

# ╔═╡ 55088862-8189-4f93-89a8-6a691e6bbce0
𝒮 = [HUNGRYₛ, FULLₛ]

# ╔═╡ 2453dca9-12ff-4f0c-88a0-b03dd0f0863e
md"### Action"

# ╔═╡ 22ef8f41-327f-4c2d-9863-ee5a7bd13751
𝒜 = [FEEDₐ, IGNOREₐ]

# ╔═╡ 9771d60d-b350-4a6a-a351-a578540d6eab
md"### Observation Space"

# ╔═╡ 6557aaef-9951-48d9-8a82-8ab646edcca1
𝒪 = [CRYINGₒ, QUIETₒ]

# ╔═╡ efd64c90-9336-4777-9048-2ccc1943c8ee
md"""### Initial State
for our initial state we define Deterministically full """

# ╔═╡ a821edbd-0f17-4825-a0fe-d92d50a127ba
initialstate_distr = Deterministic(FULLₛ);

# ╔═╡ b6dc2a43-0e14-45a8-bf7e-ae9a96ee78ef
md"""
### Transition Function
The transition dynamics are $T(s^\prime \mid s, a)$:

$$\begin{align}
T(\rm hungry \mid hungry, feed) &= 0\%\\
T(\rm full \mid hungry, feed) &= 100\%
\end{align}$$

$$\begin{align}
T(\rm hungry \mid full, feed) &= 0\%\\
T(\rm full \mid full, feed) &= 100\%
\end{align}$$

$$\begin{align}
T(\rm hungry \mid hungry, ignore)&= 100\%\\
T(\rm full \mid hungry, ignore)&= 0\%\\
\end{align}$$

$$\begin{align}
T(\rm hungry \mid full, ignore) &= 10\%\\
T(\rm full \mid full, ignore) &= 90\%
\end{align}$$

Note we include the implied complements for completeness.
"""

# ╔═╡ 57e8c25f-7552-4283-a90a-35b1fb0b7d52
function T(s::State, a::Action)
 	p_hungry::Real = params.p_become_hungry

	if a == FEEDₐ
		return SparseCat([HUNGRYₛ, FULLₛ], [0,1])
	elseif s == HUNGRYₛ && a == IGNOREₐ
		return SparseCat([HUNGRYₛ, FULLₛ], [1,0])
	elseif s == FULLₛ && a == IGNOREₐ
		return SparseCat([HUNGRYₛ, FULLₛ], [p_hungry, 1-p_hungry])
	end
end

# ╔═╡ b184547d-4fd2-467e-a6c0-707da1f7a76c
md"""
### Observation Function
The observation function, or observation model, $O(o \mid s^\prime)$ is given by:

$$\begin{align}
O(\rm crying \mid hungry) &= 80\%\\
O(\rm quiet \mid hungry) &= 20\%
\end{align}$$

$$\begin{align}
O(\rm crying \mid full) &= 10\%\\
O(\rm quiet \mid full) &= 90\%
\end{align}$$
"""

# ╔═╡ e25fef86-e2ce-47e1-a52d-27b01373455b
function O(s::State, a::Action, s′::State)
	if s′ == HUNGRYₛ
		return SparseCat([CRYINGₒ, QUIETₒ], 
						 [params.p_crying_when_hungry, 1-params.p_crying_when_hungry])
	elseif s′ == FULLₛ
		return SparseCat([CRYINGₒ, QUIETₒ],
						[params.p_crying_when_full, 1-params.p_crying_when_full])
	end
end

# ╔═╡ 688fd1d1-80b3-4cfe-a59c-bfde92629240
O(a::Action, s′::State) = O(FULLₛ, a, s′)

# ╔═╡ 3947ee50-7dbe-439e-916b-688f95dfcd68
md"""
### Reward Function
The reward function is addative, meaning we get a reward of $r_\text{hungry}$ whenever the baby is hungry *plus* $r_\text{feed}$ whenever we feed the baby.
"""

# ╔═╡ ef6e3676-4942-4fde-a8a7-0d5c39c55d04
function R(s::State, a::Action)
 	return(s == HUNGRYₛ ? params.r_hungry : 0) + (a == FEEDₐ ? params.r_feed : 0)
end

# ╔═╡ 7f98be26-f6d6-4284-820f-1aa77ddb7f79
md"
### Discount Factor
For an infinite horizon problem, we set the discount factor $\gamma \in [0,1]$ to a value where $\gamma < 1$ to discount future rewards.
"

# ╔═╡ 5d981618-28de-4007-89fd-dd197c13eb9c
γ = 0.9;

# ╔═╡ 35afd0eb-a891-47af-a076-81dbb340e516
md"## POMDP Structure"

# ╔═╡ 7bf74982-c5f4-468f-b588-dc3b8e125525
abstract type CryingBaby <: POMDP{State,Action,Observation}end

# ╔═╡ 472d5125-53a9-4936-8f82-60977d55fda6
pomdp = QuickPOMDP(CryingBaby,
				  states = 𝒮,
				  actions = 𝒜,
				  observations = 𝒪,
				  transition = T,
				  reward = R,
				  observation = O,
				  discount = γ,
				  initialstate = initialstate_distr);

# ╔═╡ 01513401-2101-4fb4-b736-cfa3e480e516
md"### Policy
We create a simple `Policy` type with an associated `POMDPs.action` function which always feeds the baby when we it's crying.

The `POMDPs.action(π, s)` function maps the current state $s$ (or belief state $b(s)$ for POMDPs) to an action $a$ given a policy $\pi$.

$$\begin{align}
\pi(s) &= a\tag{for MDPs}\\
\pi(b) &= a\tag{for POMDPs}
\end{align}$$
"

# ╔═╡ 7143c1c8-73bf-4b77-a89d-c2109ab0dff0
md"""
For the simple case, let's define a policy where we alway feed when we observe the baby crying.
"""

# ╔═╡ c5776011-e4f2-4bab-b778-156aaa550fc5
struct FeedWhenCrying <: Policy end

# ╔═╡ 6fd1399a-963c-44f8-b49c-ad8686edf53d
md"And a policy that feeds the baby when we believe it to be hungry."

# ╔═╡ e9ca2f42-614e-4d93-a9b2-a4c40d5d759a
struct FeedWhenBelievedHungry <: Policy end

# ╔═╡ d4973712-737c-46ba-98bb-3cf3e9384aff
md"### Belief"

# ╔═╡ 1f4010c8-dee0-46be-9efa-544ac213578a
const Belief = Vector{Real};

# ╔═╡ 0db34532-fd40-49b2-972e-6f8d22d09c0a
md"### Simple Policies"

# ╔═╡ ebe89856-d8a9-4b70-b129-fe497fc5e188
function POMDPs.action(::FeedWhenCrying, o::Observation)
	return o == CRYINGₒ ? FEEDₐ : IGNOREₐ
end

# ╔═╡ dce032c2-29db-4b16-abdd-f6fde5480527
function POMDPs.action(::FeedWhenBelievedHungry, b::Belief)
	return b[1] > b[2] ? FEEDₐ : IGNOREₐ
end

# ╔═╡ 0b5085c1-57e8-4207-80fa-9eb1173b9af9
md"### Belief Updater"

# ╔═╡ 542a878c-d0dd-40fc-876f-9db2fd9eca71
updater(pomdp::QuickPOMDP{CryingBaby}) = DiscreteUpdater(pomdp)

# ╔═╡ b3288b34-7781-4e11-86a2-28baa8cdaddf
md"""
We start out with a uniform belief over where $p(\texttt{hungry}) = 0.5$ and $p(\texttt{full})=0.5$.
"""

# ╔═╡ a7d6e778-d38d-4e26-bd5e-396f09d1a560
b0 = uniform_belief(pomdp); b0.b

# ╔═╡ e36cf18e-3d7a-46e0-99be-a9b866db38bc
md"""
Then we can "update" our current belief based on our selected action and subsequent observation. The `update` function has the following signature:
```julia
update(::Updater, belief_old, action, observation)
```
"""

# ╔═╡ aa701e6b-ddc2-4cd7-9bb2-7d60db9b9a29
begin
	a1 = IGNOREₐ
	o1 = CRYINGₒ
	b1 = update(updater(pomdp), b0, a1, o1)
	b1.b 
end

# ╔═╡ 01531a02-91e5-463a-be72-32f66787fcbe
begin
	a2 = FEEDₐ
	o2 = QUIETₒ
	b2 = update(updater(pomdp), b1, a2, o2)
	b2.b
end

# ╔═╡ 0e1246c7-c32e-4198-bfa5-c4d5a2717788
begin
	a3 = IGNOREₐ
	o3 = QUIETₒ
	b3 = update(updater(pomdp), b2, a3, o3)
	b3.b
end

# ╔═╡ e2c0a434-8367-4287-8692-39136d789590
begin
	a4 = IGNOREₐ
	o4 = QUIETₒ
	b4 = update(updater(pomdp), b3, a4, o4)
	b4.b
end

# ╔═╡ 16704c6e-7342-4222-be95-0b0415118928
begin
	a5 = IGNOREₐ
	o5 = CRYINGₒ
	b5 = update(updater(pomdp), b4, a5, o5)
	b5.b
end

# ╔═╡ 24c4b592-809a-4a6a-9273-d8f652d1a926
md"""
## Solutions: _Offline_
As with POMDPs, we can solve for a policy either _offline_ (to generate a full mapping from _beliefs_ to _actions_ for all _states_) or _online_ to only generate a mapping from the current belief state to the next action.

Solution methods typically follow the defined `POMDPs.jl` interface syntax:

```julia
solver = FancyAlgorithmSolver() # inputs are the parameters of said algorithm
policy = solve(solver, pomdp)   # solves the POMDP and returns a policy
```
"""

# ╔═╡ 55950a02-8c85-48ed-8fac-ec743f9ce042
md"""
### Policy Representation: Alpha Vectors
Since we do not know the current state exactly, we can compute the *utility* of our belief *b*

$$U(b) = \sum_s b(s)U(s) = \mathbf{α}^\top \mathbf{b}$$

where $\mathbf{α}$ is called an _alpha vector_ that contains the expected utility for each _belief state_ under a policy.
"""

# ╔═╡ 2cda0eeb-e7a2-453d-85eb-bb5c3182603e
md"""
### QMDP
To solve the POMDP, we first need a *solver*. We'll use the QMDP solver$^3$ from `QMDP.jl`. QMDP will treat each belief state as the true state (thus turning it into an MDP), and then use value iteration to solve that MDP.

$$\alpha_a^{(k+1)}(s) = R(s,a) + \gamma\sum_{s'}T(s'\mid s, a)\max_{a'}\alpha_{a'}^{(k)}(s')$$
"""

# ╔═╡ 25a2ecce-0b8d-49a4-8322-319e80882fd1
md"""
### Fast Informed Bound (FIB)
Another _offline_ POMDP solver is the _fast informed bound_ (FIB)$^2$. FIB actually uses information from the observation model $O$ (i.e. "informed").

$$\alpha_a^{(k+1)}(s) = R(s,a) + \gamma\sum_o\max_{a'}\sum_{s'}O(o \mid a,s')T(s'\mid s, a)\alpha_{a'}^{(k)}(s')$$

See the usage here: [https://github.com/JuliaPOMDP/FIB.jl](https://github.com/JuliaPOMDP/FIB.jl)
"""

# ╔═╡ 5dbc4fd0-cfec-48ab-9598-03d32f9cb476
fib_solver = FIBSolver()

# ╔═╡ 1945839b-ff83-4796-ae53-c991c3b10bde
fib_policy = solve(fib_solver, pomdp)

# ╔═╡ 9a56fee7-ee59-4825-8b51-fc1cab210ea1
md"""
### Point-Based Value Iteration (PBVI)
_Point-based value iteration_ provides a lower bound and operates on a finite set of $m$ beliefs $B=\{\mathbf{b}_1, \ldots, \mathbf{b}_m\}$, each with an associated alpha vector $\Gamma = \{\boldsymbol{\alpha}_1, \ldots, \boldsymbol{\alpha}_m\}$. These alpha vector define an _approximately optimal value function_:

$$U^\Gamma(\mathbf{b}) = \max_{\boldsymbol\alpha \in \Gamma}\boldsymbol\alpha^\top\mathbf{b}$$

with a lower bound on the optimal value function, $U^\Gamma(\mathbf{b}) \le U^*(\mathbf{b})$ for all $\mathbf{b}$.

PBVI iterates through every possible action $a$ and observation $o$ to extract the alpha vector from the set $\Gamma$ that is maximal at the _resulting_ (i.e., updated) belief $\mathbf{b}'$:

$$\begin{align*}
	\boldsymbol{\alpha}_{a,o} &= \operatorname*{arg\,max}_{\boldsymbol{\alpha} \in \Gamma}\boldsymbol{\alpha}^\top\operatorname{Update}(\mathbf{b}, a, o)\\
                             &= \operatorname*{arg\,max}_{\boldsymbol{\alpha} \in \Gamma}\boldsymbol{\alpha}^\top\mathbf{b}'
\end{align*}$$

Then we construct a new alpha vector for each action $a$ based on these $\boldsymbol{\alpha}_{a,o}$ vectors:

$$\alpha_a(s) = R(s,a) + \gamma\sum_{s',o}O(o \mid a,s')T(s'\mid s, a)\alpha_{a,o}(s')$$

With the final alpha vector produced by the backup operator being:

$$𝛂 = \operatorname*{arg\,max}_{𝛂_a} 𝛂_a^\top \mathbf{b}$$
"""

# ╔═╡ ce918362-93bf-460a-b4b1-24585ce14974
pbvi_solver = PBVISolver()

# ╔═╡ 8b337757-cce0-4dfb-9569-819e4590736c
pbvi_policy = solve(pbvi_solver, pomdp)

# ╔═╡ 20b673be-9947-465c-8ac6-b17e87aa6c8d
md"""
## Visualizing Alpha Vectors
**_Recall_**: Since we do not know the current state exactly, we can compute the *utility* of our belief *b*

$$U(b) = \sum_s b(s)U(s) = \mathbf{α}^\top \mathbf{b}$$

where $\mathbf{α}$ is called an _alpha vector_ that contains the expected utility for each _belief state_ under a policy.
"""

# ╔═╡ 69069a49-a254-48d8-9874-ae8703186f24
function plot_alpha_vectors(policy, p_hungry, label="QMDP")
	# calculate the maximum utility, which determines the action to take
	current_belief = [p_hungry, 1-p_hungry]
	feed_idx = Int(policy.action_map[1])+1
	ignore_idx = Int(policy.action_map[2])+1
	utility_feed = policy.alphas[feed_idx]' * current_belief # dot product
	utility_ignore = policy.alphas[ignore_idx]' * current_belief # dot product
	lw_feed, lw_ignore = 1, 1
	check_feed, check_ignore = "", ""
	if utility_feed >= utility_ignore
		current_utility = utility_feed
		lw_feed = 2
		check_feed = "✓"
	else
		current_utility = utility_ignore
		lw_ignore = 2
		check_ignore = "✓"
	end
	
	# plot the alpha vector hyperplanes
	plot(size=(600,340))
	plot!(Int.([FULLₛ, HUNGRYₛ]), policy.alphas[ignore_idx],
		  label="ignore ($label) $(check_ignore)", c=:red, lw=lw_ignore)
	plot!(Int.([FULLₛ, HUNGRYₛ]), policy.alphas[feed_idx],
		  label="feed ($label) $(check_feed)", c=:blue, lw=lw_feed)
	
	# plot utility of selected action
	rnd(x) = round(x,digits=3)
	scatter!([p_hungry], [current_utility], 
		     c=:black, ms=5, label="($(rnd(p_hungry)), $(rnd(current_utility)))")

	title!("Alpha Vectors")
	xlabel!("𝑝(hungry)")
	ylabel!("utility 𝑈(𝐛)")
	xlims!(0, 1)
	ylims!(-40, 5)
end

# ╔═╡ 79be3f8c-5f7f-4f58-b256-e425d32a434e
md"### QMDP Alpha Vectors"

# ╔═╡ 93b4da42-64a8-42b5-b6d3-8ead97e4c54a
@bind p_hungry Slider(0:0.01:1, default=0.5, show_value=true)

# ╔═╡ 1d1d4ef1-4f09-476f-9946-f1ba45bba7d5
[p_hungry, 1-p_hungry]

# ╔═╡ 0f078a9e-d0d4-408b-a597-75fca52f21dc
@bind qmdp_iters Slider(0:60, default=60, show_value=true)

# ╔═╡ 9f643d17-7c28-4915-be7a-0e515769b2e1
qmdp_solver = QMDPSolver(max_iterations=qmdp_iters);

# ╔═╡ adfc93a1-5876-4f5b-bfb3-3cf83f426248
qmdp_policy = solve(qmdp_solver, pomdp)

# ╔═╡ adc6fbbc-cd57-4f42-a90d-da4273efb11e
plot_alpha_vectors(qmdp_policy, p_hungry)

# ╔═╡ bea823a4-c9a1-4ab7-b9bc-3966d95f91bf
action(qmdp_policy, [p_hungry, 1-p_hungry])

# ╔═╡ 45cf95a3-2756-464c-a2da-7b406dee78f6
qmdp_policy.alphas

# ╔═╡ 646e1cb7-9044-497a-97e6-0821bbf4e912
md"### PBVI Alpha Vectors"

# ╔═╡ c233c813-9f64-4b22-b155-b630679e263b
@bind p Slider(0:0.01:1, default=0.5, show_value=true)

# ╔═╡ 20b52466-bde4-4e0f-85af-462befde0a6b
plot_alpha_vectors(pbvi_policy, p, "PBVI")

# ╔═╡ ddff21da-12e8-4851-8fcc-89b35937dfcc
𝐛 = [p, 1-p]

# ╔═╡ 9fa2365d-7fcd-4c88-b844-a1a76e54af9a
action(pbvi_policy, 𝐛)

# ╔═╡ d05fc548-b0c1-4117-863f-1dfd10b88f42
𝛂 = argmax(αₐ->αₐ'*𝐛, pbvi_policy.alphas)

# ╔═╡ 298a09c4-22e3-4b6a-8be7-bb03a14ff1dc
pbvi_policy.alphas

# ╔═╡ e65eede9-4278-44f3-a2ce-9135989684d7
md"### FIB Alpha Vectors"

# ╔═╡ 3274825e-6c1f-41cf-946b-e90a2ce68c39
@bind fib_hungry Slider(0:0.01:1, default=0.5, show_value=true)

# ╔═╡ 574e4fbc-c491-4f2f-8676-de6567c68e87
plot_alpha_vectors(fib_policy, fib_hungry, "FIB")

# ╔═╡ 9645f45d-8d90-44f0-98d5-94ab99fda365
fib_b = [fib_hungry, 1- fib_hungry]

# ╔═╡ 6ca48934-d960-49f2-bec3-1d9ae8871fbb
action(fib_policy, fib_b)

# ╔═╡ 66f6c6e2-de25-4ac0-ae67-e246a6e9b5f4
fib_policy.alphas

# ╔═╡ 59b888b2-4893-4f10-800f-1ce7de7f20fb
md"### Dominant Alpha Vectors"

# ╔═╡ df3dc07d-67da-4343-af9b-940e0c9ddec2
@bind show_thresholds CheckBox(true)

# ╔═╡ 1dce45b3-7191-4e7e-861c-fb843924a100
@bind show_fib CheckBox(true)

# ╔═╡ ff6915c0-b007-49d0-b63e-bbeba4848663
@bind show_pbvi CheckBox(true)

# ╔═╡ 92d9c106-78b3-4d63-a766-35fb67be1265
begin
	p_range = 0:0.001:1

	dominating_action_idx(policy, 𝐛) = Int(action(policy, 𝐛))+1

	dominant_actions(policy) = map(p->
		dominating_action_idx(policy,[p,1-p]), p_range)

	dominant_line(policy) = map(p->
		policy.alphas[dominating_action_idx(policy,[p,1-p])]'*[p,1-p], p_range)

	dominant_line_multiple_α(policy) = map(p->
		argmax(αₐ->αₐ'*[p,1-p], policy.alphas)'*[p,1-p], p_range)

	dominant_color(policy, c1=:blue, c2=:red) = map(p->
		dominating_action_idx(policy,[p,1-p]) == 1 ? c1 : c2, p_range)

	qmdp_solver2 = QMDPSolver()
	qmdp_policy2 = solve(qmdp_solver2, pomdp)

	fib_solver2 = FIBSolver()
	fib_policy2 = solve(fib_solver2, pomdp)

	pbvi_solver2 = PBVISolver()
	pbvi_policy2 = solve(pbvi_solver2, pomdp)

	dominant_line_qmdp = dominant_line(qmdp_policy2)
	dominant_color_qmdp = dominant_color(qmdp_policy2)

	dominant_line_fib = dominant_line(fib_policy2)
	dominant_color_fib = dominant_color(fib_policy2, :cyan, :magenta)

	dominant_line_pbvi = dominant_line_multiple_α(pbvi_policy2)
	dominant_color_pbvi = dominant_color(pbvi_policy2, :green, :black)
	
	# plot the dominant alpha vector hyperplanes
	plot(size=(600,340))
	plot!(p_range, dominant_line_qmdp, label="QMDP",
		  c=dominant_color_qmdp, lw=2)
	
	if show_fib
		plot!(p_range, dominant_line_fib, label="FIB",
			  c=dominant_color_fib, lw=2)
	end
	
	if show_pbvi
		plot!(p_range, dominant_line_pbvi, label="PBVI",
			  c=dominant_color_pbvi, lw=2)
	end

	thresh_qmdp = p_range[findfirst(dominant_actions(qmdp_policy2) .== 1)]
	thresh_fib = p_range[findfirst(dominant_actions(fib_policy2) .== 1)]
	thresh_pbvi = p_range[findfirst(dominant_actions(pbvi_policy2) .== 1)]
	
	if show_thresholds
		plot!([thresh_qmdp, thresh_qmdp], [-40, 5], color=:gray, style=:dash,
			  label="p ≈ $thresh_qmdp (QMDP)")

		if show_fib
			plot!([thresh_fib, thresh_fib], [-40, 5], color=:gray, style=:dash,
				  label="p ≈ $thresh_fib (FIB)")
		end

		if show_pbvi
			plot!([thresh_pbvi, thresh_pbvi], [-40, 5], color=:gray, style=:dash,
				  label="p ≈ $thresh_pbvi (PBVI)")
		end
	end

	title!("Dominant Alpha Vectors")
	xlabel!("𝑝(hungry)")
	ylabel!("utility 𝑈(𝐛)")
	xlims!(0, 1)
	ylims!(-40, 5)
end

# ╔═╡ 994722fd-b8ba-4ff8-bd47-2eb015c8c5b3
begin
	dominant_color_pbvi2 = dominant_color(pbvi_policy2)
	
	# plot the dominant alpha vector hyperplanes
	plot(size=(600,340))
	plot!(p_range, dominant_line_pbvi, label="PBVI",
		  c=dominant_color_pbvi2, lw=2)

	title!("Dominant Alpha Vectors")
	xlabel!("𝑝(hungry)")
	ylabel!("utility 𝑈(𝐛)")
	xlims!(0, 1)
	ylims!(-40, 5)
end

# ╔═╡ d4a22895-fb4e-45f1-a799-9b26bd96a805
md"## Solutions: Online"

# ╔═╡ 7662add3-c973-4807-a2a0-4bd63995e8fc
md"### Partially Observable Monte Carlo Planning"

# ╔═╡ 6b4c780a-fd64-463a-9069-a587133d2d5d
pomcp_solver = POMCPSolver()

# ╔═╡ 53483b89-c77c-4a1a-bd55-495458e9ce70
pomcp_planner = solve(pomcp_solver, pomdp)

# ╔═╡ c9d66b2a-c89d-46ec-a428-d0d0c96ff7bb
initialstate(pomdp)

# ╔═╡ 7bd0c88c-8884-4f80-a017-6abaa0933912
aₚ, info = action_info(pomcp_planner, initialstate(pomdp), tree_in_info=true);aₚ

# ╔═╡ e09ab314-38da-401a-a5a9-2027a1254e60
tree = D3Tree(info[:tree], init_expand=3)

# ╔═╡ 6ac6f3e2-46b8-4929-8d9e-8ea21957fcd5
md"""
## Concise POMDP definition

```julia
using POMDPs, POMDPModelTools, QuickPOMDPs

@enum State hungry full
@enum Action feed ignore
@enum Observation crying quiet

pomdp = QuickPOMDP(
    states       = [hungry, full],  # 𝒮
    actions      = [feed, ignore],  # 𝒜
    observations = [crying, quiet], # 𝒪
    initialstate = [full],          # Deterministic initial state
    discount     = 0.9,             # γ

    transition = function T(s, a)
        if a == feed
            return SparseCat([hungry, full], [0, 1])
        elseif s == hungry && a == ignore
            return SparseCat([hungry, full], [1, 0])
        elseif s == full && a == ignore
            return SparseCat([hungry, full], [0.1, 0.9])
        end
    end,

    observation = function O(s, a, s′)
        if s′ == hungry
            return SparseCat([crying, quiet], [0.8, 0.2])
        elseif s′ == full
            return SparseCat([crying, quiet], [0.1, 0.9])
        end
    end,

    reward = (s,a)->(s == hungry ? -10 : 0) + (a == feed ? -5 : 0)
)

# Solve POMDP
using QMDP
solver = QMDPSolver()
policy = solve(solver, pomdp)

# Query policy for an action, given a belief vector
𝐛 = [0.2, 0.8]
a = action(policy, 𝐛)
```
"""

# ╔═╡ c3054823-488f-4a35-b66f-2469abfd39e4
TableOfContents(title="HandsOn POMDP", depth=4)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BasicPOMCP = "d721219e-3fc6-5570-a8ef-e5402f47c49e"
BeliefUpdaters = "8bb6e9a1-7d73-552c-a44a-e5dc5634aac4"
D3Trees = "e3df1716-f71e-5df9-9e2d-98e193103c45"
FIB = "13b007ba-0ca8-5af2-9adf-bc6a6301e25a"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
POMDPModelTools = "08074719-1b2a-587c-a292-00f91cc44415"
POMDPPolicies = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PointBasedValueIteration = "835c131e-675f-4498-8e2c-c054c75556e1"
QMDP = "3aa3ecc9-5a5d-57c8-8188-3e47bd8068d2"
QuickPOMDPs = "8af83fb2-a731-493c-9049-9e19dbce6165"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BasicPOMCP = "~0.3.12"
BeliefUpdaters = "~0.2.3"
D3Trees = "~0.3.8"
FIB = "~0.4.5"
JSON = "~1.6.1"
POMDPModelTools = "~0.3.13"
POMDPPolicies = "~0.4.3"
POMDPs = "~0.9.6"
Parameters = "~0.12.3"
Plots = "~1.41.6"
PlutoUI = "~0.7.83"
PointBasedValueIteration = "~0.2.4"
QMDP = "~0.1.8"
QuickPOMDPs = "~0.2.15"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.11"
manifest_format = "2.0"
project_hash = "e6fb61cebd2a8088eaf8baf8fb0c5a9a8877d63d"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BasicPOMCP]]
deps = ["Colors", "D3Trees", "MCTS", "POMDPLinter", "POMDPTools", "POMDPs", "Parameters", "ParticleFilters", "Printf", "Random"]
git-tree-sha1 = "90d7dfc64ebdee34659a400892c011c3f4eea46e"
uuid = "d721219e-3fc6-5570-a8ef-e5402f47c49e"
version = "0.3.12"

[[deps.BeliefUpdaters]]
deps = ["POMDPTools", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "8819a9a0e9e9002125ae55626e10f0c210959c30"
uuid = "8bb6e9a1-7d73-552c-a44a-e5dc5634aac4"
version = "0.2.3"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonRLInterface]]
deps = ["Tricks"]
git-tree-sha1 = "6c7d1ebb157fdf0f696698ef01946fe93c9efff4"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.3"

[[deps.CommonSolve]]
git-tree-sha1 = "99ee296f88c12485402e37c2fd025f95ae097637"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.9"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.D3Trees]]
deps = ["AbstractTrees", "HTTP", "JSON", "Random", "Sockets"]
git-tree-sha1 = "b13921e8e5bac9298df1720a56263a053fb606d3"
uuid = "e3df1716-f71e-5df9-9e2d-98e193103c45"
version = "0.3.8"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiscreteValueIteration]]
deps = ["LinearAlgebra", "POMDPLinter", "POMDPTools", "POMDPs", "Printf", "SparseArrays"]
git-tree-sha1 = "4210976635e65e2410aa7662203c1d2090320891"
uuid = "4b033969-44f6-5439-a48b-c11fa3648068"
version = "0.4.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "cd3c5ac74cd3923c8945c6a81518c46abd0e73a3"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.129"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FIB]]
deps = ["POMDPTools", "POMDPs", "Printf"]
git-tree-sha1 = "2a51ded195d93974975a3448df7df37a675081a4"
uuid = "13b007ba-0ca8-5af2-9adf-bc6a6301e25a"
version = "0.4.5"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteHorizonPOMDPs]]
deps = ["POMDPTools", "POMDPs", "Random"]
git-tree-sha1 = "bc448a46c009a58e301e198fb7642842f2010974"
uuid = "8a13bbfe-798e-11e9-2f1c-eba9ee5ef093"
version = "0.4.1"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f954322d5de03ec630d177cda203dcd92b6be399"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.26"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "6fada551286ab6ea4ca1628cb2de9f166a2ec966"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.26+0"

[[deps.Gamma]]
git-tree-sha1 = "86f86b6168a016ed88e4ae4e64577b98c3b59e8e"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.1.0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "18d7deab5fb0440dc6a7b6993c5c27b25420de10"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.29"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MCTS]]
deps = ["Colors", "D3Trees", "POMDPLinter", "POMDPTools", "POMDPs", "Printf", "ProgressMeter", "Random"]
git-tree-sha1 = "54caa2378219886966d259af1f22a8ffc6bd13b2"
uuid = "e12ccd36-dcad-5f33-8774-9175229e7b33"
version = "0.5.7"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.MarchingCubes]]
deps = ["PrecompileTools", "StaticArrays"]
git-tree-sha1 = "0e893025924b6becbae4109f8020ac0e12674b01"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.12.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+5"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d8cce34295c55f47be683580f44791716045b8fe"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.7+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "863f15ca84aec9015e9954ac0fd56ce93a5dd745"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.3"

[[deps.POMDPModelTools]]
deps = ["CommonRLInterface", "Distributions", "LinearAlgebra", "POMDPLinter", "POMDPTools", "POMDPs", "Random", "Reexport", "SparseArrays", "Statistics", "Tricks", "UnicodePlots"]
git-tree-sha1 = "36d32d62e036ae8ebb9b8efe9e8658f902815700"
uuid = "08074719-1b2a-587c-a292-00f91cc44415"
version = "0.3.13"

[[deps.POMDPPolicies]]
deps = ["Distributions", "LinearAlgebra", "POMDPTools", "POMDPs", "Parameters", "Random", "Reexport", "SparseArrays", "StatsBase"]
git-tree-sha1 = "bd72fbfea89a64946963518aa53097e2a1233c59"
uuid = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
version = "0.4.3"

[[deps.POMDPTools]]
deps = ["CommonRLInterface", "DataFrames", "Distributed", "Distributions", "LinearAlgebra", "NamedTupleTools", "POMDPLinter", "POMDPs", "Parameters", "ProgressMeter", "Random", "Reexport", "SparseArrays", "Statistics", "StatsBase", "Tricks", "UnicodePlots"]
git-tree-sha1 = "6b7e405f2c1905aff6f07ee4d241fd608d289d66"
uuid = "7588e00f-9cae-40de-98dc-e0c70c48cdd7"
version = "0.1.6"

[[deps.POMDPs]]
deps = ["Distributions", "Graphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "9a6fe01a75a23cfb8a4d7af43f95ff3db16694f1"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.6"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.ParticleFilters]]
deps = ["AliasTables", "LinearAlgebra", "POMDPLinter", "POMDPTools", "POMDPs", "Random", "ReadOnlyArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "b5bf203b7725368c2a8ad5996ab9db510082a62b"
uuid = "c8b314e2-9260-5cf8-ae76-3be7461ca6d0"
version = "0.6.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PointBasedValueIteration]]
deps = ["Distributions", "FiniteHorizonPOMDPs", "LinearAlgebra", "POMDPLinter", "POMDPTools", "POMDPs"]
git-tree-sha1 = "1d9321a3ab42c1532db4985c0e983208d4c7a990"
uuid = "835c131e-675f-4498-8e2c-c054c75556e1"
version = "0.2.4"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QMDP]]
deps = ["DiscreteValueIteration", "POMDPLinter", "POMDPTools", "POMDPs", "Random"]
git-tree-sha1 = "b091e80ccfe13370838973a9ba87bf77807d0baa"
uuid = "3aa3ecc9-5a5d-57c8-8188-3e47bd8068d2"
version = "0.1.8"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.QuickPOMDPs]]
deps = ["NamedTupleTools", "POMDPTools", "POMDPs", "Random", "Tricks", "UUIDs"]
git-tree-sha1 = "eaba1073436ee7d05605fb6958202249130f6366"
uuid = "8af83fb2-a731-493c-9049-9e19dbce6165"
version = "0.2.15"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.ReadOnlyArrays]]
git-tree-sha1 = "e6f7ddf48cf141cb312b078ca21cb2d29d0dc11d"
uuid = "988b38a3-91fc-5605-94a2-ee2116b3bd83"
version = "0.2.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "ed45bcc7cf3c8887595b973f2b1efbe91dcc50ec"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.1"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "770240df9a3b8888065046948f7a09b4e0f997d5"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "0f38a06c83f0007bbab3cf911262841c9a0f07e0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.13.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["Contour", "Crayons", "Dates", "LinearAlgebra", "MarchingCubes", "NaNMath", "SparseArrays", "StaticArrays", "StatsBase"]
git-tree-sha1 = "66f9127e995e4eab4041c5f01d644a7278ac8bc2"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "2.8.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.6.1+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╠═b3619d30-78a2-11f1-068c-7974b7ab15b3
# ╟─d6a04964-52b3-4f5e-a00d-fd3153cfc3fd
# ╟─f1d195d6-5776-4056-95b4-d4392f5ef9f9
# ╠═5050d20c-d41c-440e-a78f-9b4d644434c4
# ╠═9e6f9788-f932-4f7a-9aa3-4c8def87e8b0
# ╠═fc2971ca-bfdd-40f9-ac50-22681036e520
# ╠═bec4bdb1-b9cf-44df-8c20-74e75f8d0e5d
# ╠═f49ad3a7-8da0-4723-9f81-23d6fe3f5c45
# ╠═5adef0b9-fe46-402e-bbbc-29a460b13831
# ╠═c7664ad5-0a94-4813-b8d1-008369b50d78
# ╠═55088862-8189-4f93-89a8-6a691e6bbce0
# ╠═2453dca9-12ff-4f0c-88a0-b03dd0f0863e
# ╠═22ef8f41-327f-4c2d-9863-ee5a7bd13751
# ╠═9771d60d-b350-4a6a-a351-a578540d6eab
# ╠═6557aaef-9951-48d9-8a82-8ab646edcca1
# ╠═efd64c90-9336-4777-9048-2ccc1943c8ee
# ╠═a821edbd-0f17-4825-a0fe-d92d50a127ba
# ╟─b6dc2a43-0e14-45a8-bf7e-ae9a96ee78ef
# ╠═57e8c25f-7552-4283-a90a-35b1fb0b7d52
# ╟─b184547d-4fd2-467e-a6c0-707da1f7a76c
# ╠═e25fef86-e2ce-47e1-a52d-27b01373455b
# ╠═688fd1d1-80b3-4cfe-a59c-bfde92629240
# ╠═3947ee50-7dbe-439e-916b-688f95dfcd68
# ╠═ef6e3676-4942-4fde-a8a7-0d5c39c55d04
# ╟─7f98be26-f6d6-4284-820f-1aa77ddb7f79
# ╠═5d981618-28de-4007-89fd-dd197c13eb9c
# ╠═35afd0eb-a891-47af-a076-81dbb340e516
# ╠═7bf74982-c5f4-468f-b588-dc3b8e125525
# ╠═472d5125-53a9-4936-8f82-60977d55fda6
# ╠═01513401-2101-4fb4-b736-cfa3e480e516
# ╠═e6ad5876-cf5e-453a-96b6-bd15996a7fdb
# ╟─7143c1c8-73bf-4b77-a89d-c2109ab0dff0
# ╠═c5776011-e4f2-4bab-b778-156aaa550fc5
# ╟─6fd1399a-963c-44f8-b49c-ad8686edf53d
# ╠═e9ca2f42-614e-4d93-a9b2-a4c40d5d759a
# ╠═d4973712-737c-46ba-98bb-3cf3e9384aff
# ╠═1f4010c8-dee0-46be-9efa-544ac213578a
# ╠═0db34532-fd40-49b2-972e-6f8d22d09c0a
# ╠═ebe89856-d8a9-4b70-b129-fe497fc5e188
# ╠═dce032c2-29db-4b16-abdd-f6fde5480527
# ╠═0b5085c1-57e8-4207-80fa-9eb1173b9af9
# ╠═542a878c-d0dd-40fc-876f-9db2fd9eca71
# ╟─b3288b34-7781-4e11-86a2-28baa8cdaddf
# ╠═a7d6e778-d38d-4e26-bd5e-396f09d1a560
# ╟─e36cf18e-3d7a-46e0-99be-a9b866db38bc
# ╠═aa701e6b-ddc2-4cd7-9bb2-7d60db9b9a29
# ╠═01531a02-91e5-463a-be72-32f66787fcbe
# ╠═0e1246c7-c32e-4198-bfa5-c4d5a2717788
# ╠═e2c0a434-8367-4287-8692-39136d789590
# ╠═16704c6e-7342-4222-be95-0b0415118928
# ╟─24c4b592-809a-4a6a-9273-d8f652d1a926
# ╟─55950a02-8c85-48ed-8fac-ec743f9ce042
# ╟─2cda0eeb-e7a2-453d-85eb-bb5c3182603e
# ╠═f83fa2ae-5532-45d5-86d2-e2bb26e47285
# ╠═9f643d17-7c28-4915-be7a-0e515769b2e1
# ╠═adfc93a1-5876-4f5b-bfb3-3cf83f426248
# ╟─25a2ecce-0b8d-49a4-8322-319e80882fd1
# ╠═c542c586-bb71-4509-9c5c-5d5870cb53e3
# ╠═5dbc4fd0-cfec-48ab-9598-03d32f9cb476
# ╠═1945839b-ff83-4796-ae53-c991c3b10bde
# ╟─9a56fee7-ee59-4825-8b51-fc1cab210ea1
# ╠═6e1c8f95-4c62-4b97-b6b8-f76455a867e1
# ╠═ce918362-93bf-460a-b4b1-24585ce14974
# ╠═8b337757-cce0-4dfb-9569-819e4590736c
# ╟─20b673be-9947-465c-8ac6-b17e87aa6c8d
# ╠═8ed40b93-2bc4-4e2a-87c0-5936516813d4
# ╠═f8a3389c-3cec-4438-98cc-34b0a70025d3
# ╠═af668bdd-adab-48b0-ba41-c42b49d5c39b
# ╠═69069a49-a254-48d8-9874-ae8703186f24
# ╠═79be3f8c-5f7f-4f58-b256-e425d32a434e
# ╠═adc6fbbc-cd57-4f42-a90d-da4273efb11e
# ╠═1d1d4ef1-4f09-476f-9946-f1ba45bba7d5
# ╠═bea823a4-c9a1-4ab7-b9bc-3966d95f91bf
# ╠═93b4da42-64a8-42b5-b6d3-8ead97e4c54a
# ╠═0f078a9e-d0d4-408b-a597-75fca52f21dc
# ╠═45cf95a3-2756-464c-a2da-7b406dee78f6
# ╠═646e1cb7-9044-497a-97e6-0821bbf4e912
# ╠═20b52466-bde4-4e0f-85af-462befde0a6b
# ╠═c233c813-9f64-4b22-b155-b630679e263b
# ╠═ddff21da-12e8-4851-8fcc-89b35937dfcc
# ╠═9fa2365d-7fcd-4c88-b844-a1a76e54af9a
# ╠═d05fc548-b0c1-4117-863f-1dfd10b88f42
# ╠═298a09c4-22e3-4b6a-8be7-bb03a14ff1dc
# ╠═e65eede9-4278-44f3-a2ce-9135989684d7
# ╠═574e4fbc-c491-4f2f-8676-de6567c68e87
# ╠═3274825e-6c1f-41cf-946b-e90a2ce68c39
# ╠═9645f45d-8d90-44f0-98d5-94ab99fda365
# ╠═6ca48934-d960-49f2-bec3-1d9ae8871fbb
# ╠═66f6c6e2-de25-4ac0-ae67-e246a6e9b5f4
# ╠═59b888b2-4893-4f10-800f-1ce7de7f20fb
# ╠═df3dc07d-67da-4343-af9b-940e0c9ddec2
# ╠═1dce45b3-7191-4e7e-861c-fb843924a100
# ╠═ff6915c0-b007-49d0-b63e-bbeba4848663
# ╠═92d9c106-78b3-4d63-a766-35fb67be1265
# ╠═994722fd-b8ba-4ff8-bd47-2eb015c8c5b3
# ╠═d4a22895-fb4e-45f1-a799-9b26bd96a805
# ╠═7662add3-c973-4807-a2a0-4bd63995e8fc
# ╠═7bb6ea08-5ed2-4706-9b51-06b88106ea64
# ╠═6b4c780a-fd64-463a-9069-a587133d2d5d
# ╠═53483b89-c77c-4a1a-bd55-495458e9ce70
# ╠═c9d66b2a-c89d-46ec-a428-d0d0c96ff7bb
# ╠═7bd0c88c-8884-4f80-a017-6abaa0933912
# ╠═58a8e286-3537-4b10-96b5-7a294b2f3b08
# ╠═b2c89f40-d530-4dca-bb25-76f85028506e
# ╠═e09ab314-38da-401a-a5a9-2027a1254e60
# ╟─6ac6f3e2-46b8-4929-8d9e-8ea21957fcd5
# ╠═c3054823-488f-4a35-b66f-2469abfd39e4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
