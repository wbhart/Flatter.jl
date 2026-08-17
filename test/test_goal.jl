# test/test_goal.jl
#
# Assumes `using Test` and `using Random` from runtests.jl.
#
# The goal is pure arithmetic, so most of this is exact-value and round-trip
# checking. The interesting part is `goal_check`, whose three heuristic
# conditions each have to be shown to do work an easier test would miss.

const GOAL_BKZ = 0.031281
const GOAL_HERMITE = 0.41503749927884365

goal_flat_profile(n, value = 5.0) = fill(Float64(value), n)
goal_linear_profile(n, top, slope) = [Float64(top) - slope * (i - 1) for i in 1:n]

@testset "reduction goal" begin

    @testset "constants match flatter's goal.h" begin
        @test Flatter.BKZ_BEST_SLOPE == GOAL_BKZ
        @test Flatter.HERMITE_BEST_SLOPE == GOAL_HERMITE
        @test Flatter.DEFAULT_G == 3.0
    end

    @testset "profile drop and spread" begin
        # Spread is the raw range.
        @test Flatter.profile_spread([10.0, 8.0, 6.0, 4.0]) ≈ 6.0
        @test Flatter.profile_spread([10.0, 20.0, 1.0, 7.0]) ≈ 19.0
        @test Flatter.profile_spread([1.0, 5.0]) ≈ 4.0

        # Drop discounts clean upward gaps, so on a decreasing profile — which
        # has none — it equals the spread.
        @test Flatter.profile_drop([10.0, 8.0, 6.0, 4.0]) ≈ 6.0
        @test Flatter.profile_drop([10.0, 20.0, 1.0, 7.0]) ≈ 19.0
        @test Flatter.profile_drop([100.0, 99.0, 10.0, 9.0]) ≈ 91.0

        # A strictly increasing profile is all gap, so its drop is zero.
        @test Flatter.profile_drop([1.0, 5.0]) ≈ 0.0
        @test Flatter.profile_drop([1.0, 5.0, 9.0]) ≈ 0.0

        # A single upward gap in the middle is discounted, the rest is not.
        # [10, 9 | 30, 29]: spread 21, gap 19, leaving the two blocks' own
        # extents of 1 each.
        @test Flatter.profile_spread([10.0, 9.0, 30.0, 29.0]) ≈ 21.0
        @test Flatter.profile_drop([10.0, 9.0, 30.0, 29.0]) ≈ 2.0

        @test Flatter.profile_drop(Float64[]) == 0.0
        @test Flatter.profile_drop([3.0]) == 0.0
        @test Flatter.profile_drop(fill(7.0, 5)) ≈ 0.0

        @testset "0 <= drop <= spread, and drop == spread when decreasing" begin
            rng = MersenneTwister(0xD40B)
            for _ in 1:2000
                n = rand(rng, 1:12)
                prof = (rand(rng, n) .- 0.5) .* 100
                drop = Flatter.profile_drop(prof)
                spread = Flatter.profile_spread(prof)
                @test drop >= -1e-9
                @test drop <= spread + 1e-9

                descending = sort(prof; rev = true)
                @test Flatter.profile_drop(descending) ≈ spread rtol = 1e-10

                n >= 2 || continue
                ascending = sort(prof)
                if allunique(prof)
                    @test Flatter.profile_drop(ascending) ≈ 0.0 atol = 1e-9
                end
            end
        end
    end

    @testset "the shape function" begin
        # 3 * (1 + 3^(log2 k + 1) - 2^(log2 k + 2)) / 2 at exact powers of two.
        @test Flatter._goal_shape(1) ≈ 0.0 atol = 1e-9
        @test Flatter._goal_shape(2) ≈ 3.0 atol = 1e-9
        @test Flatter._goal_shape(4) ≈ 18.0 atol = 1e-9
        @test Flatter._goal_shape(8) ≈ 75.0 atol = 1e-9

        # Closed form used in the implementation.
        for k in 1:200
            expected = 3 * (1 + 3.0^(log2(k) + 1) - 2.0^(log2(k) + 2)) / 2
            @test Flatter._goal_shape(k) ≈ expected atol = 1e-8 rtol = 1e-12
        end

        # Strictly increasing beyond k = 1, which is what makes a subgoal's
        # budget smaller than its parent's.
        @test all(Flatter._goal_shape(k) < Flatter._goal_shape(k + 1) for k in 1:100)
    end

    @testset "slope conversions" begin
        @testset "max_drop / n recovers the slope exactly" begin
            for n in (2, 3, 4, 5, 8, 13, 16, 64, 257),
                slope in (0.04, 0.05, 0.1, 0.3, 0.9)
                goal = Flatter.goal_from_slope(n, slope)
                @test Flatter.goal_max_drop(goal) / n ≈ slope rtol = 1e-12
            end
        end

        @testset "slopes below the BKZ floor are clamped" begin
            for slope in (-1.0, 0.0, 0.01, GOAL_BKZ)
                goal = Flatter.goal_from_slope(8, slope)
                @test Flatter.goal_max_drop(goal) / 8 ≈ GOAL_BKZ rtol = 1e-12
            end
            # Just above the floor is not clamped.
            goal = Flatter.goal_from_slope(8, GOAL_BKZ + 0.01)
            @test Flatter.goal_max_drop(goal) / 8 ≈ GOAL_BKZ + 0.01 rtol = 1e-12
        end

        @testset "root Hermite factor round trip" begin
            # Above the clamp the round trip is exact.
            for n in (4, 16, 64), rhf in (1.02, 1.05, 1.1)
                goal = Flatter.goal_from_rhf(n, rhf)
                @test Flatter.goal_rhf(goal) ≈ rhf rtol = 1e-12
            end
            # Below it, every request collapses to the same goal: asking for a
            # better factor than BKZ achieves does not make the goal stricter.
            floor_rhf = 2.0^(GOAL_BKZ / 2)
            for rhf in (1.001, 1.005, 1.01)
                @test Flatter.goal_rhf(Flatter.goal_from_rhf(16, rhf)) ≈ floor_rhf rtol = 1e-12
            end
        end

        @testset "drop conversion uses one slope per gap" begin
            n, drop = 17, 8.0
            goal = Flatter.goal_from_drop(n, drop)
            @test Flatter.goal_max_drop(goal) / n ≈ drop / (n - 1) rtol = 1e-12
        end

        @testset "dimension one has no budget to scale" begin
            goal = Flatter.goal_from_slope(1, 0.5)
            @test Flatter.goal_quality(goal) == 0.0
            @test isfinite(Flatter.goal_max_drop(goal))
            @test Flatter.goal_check(goal, [42.0])
        end
    end

    @testset "proved goals" begin
        @testset "alpha interpolates between the slopes" begin
            goal = Flatter.proved_goal(64, 0.5; base_slope = GOAL_BKZ, top_N = 64)
            @test Flatter.goal_alpha(goal, 0) ≈ GOAL_BKZ rtol = 1e-12
            @test Flatter.goal_alpha(goal, 64) ≈ 0.5 rtol = 1e-12
            # Monotone in between.
            values = [Flatter.goal_alpha(goal, k) for k in 0:64]
            @test all(values[i] < values[i + 1] for i in 1:64)
        end

        @testset "max_drop is the slope at its own dimension times n" begin
            goal = Flatter.proved_goal(32, 0.4)
            @test Flatter.goal_max_drop(goal) ≈ Flatter.goal_alpha(goal, 32) * 32 rtol = 1e-12
            @test Flatter.goal_slope(goal) ≈ 0.4 rtol = 1e-12
        end

        @testset "acceptance is the drop condition alone" begin
            goal = Flatter.proved_goal(16, 0.4)
            budget = Flatter.goal_max_drop(goal)
            @test Flatter.goal_check(goal, goal_flat_profile(16))
            # A profile whose entire drop is one step still passes, unlike the
            # heuristic goal, as long as the total is within budget.
            step = [budget * 0.9; zeros(15)]
            @test Flatter.goal_check(goal, step)
            @test !Flatter.goal_check(goal, [budget * 1.1; zeros(15)])
        end

        @testset "top_slope must exceed base_slope" begin
            @test_throws ArgumentError Flatter.proved_goal(16, 0.01)
            @test_throws ArgumentError Flatter.proved_goal(16, GOAL_BKZ)
        end
    end

    @testset "subgoal" begin
        @testset "heuristic subgoals keep quality and base slope" begin
            parent = Flatter.goal_from_slope(64, 0.1)
            child = Flatter.subgoal(parent, 0, 32)
            @test child.n == 32
            @test child.quality == parent.quality
            @test child.best_slope == parent.best_slope
            @test child.proved == false
        end

        @testset "a subgoal's budget is smaller than its parent's" begin
            for n in (4, 8, 16, 64)
                parent = Flatter.goal_from_slope(n, 0.1)
                for width in 2:(n - 1)
                    child = Flatter.subgoal(parent, 0, width)
                    @test Flatter.goal_max_drop(child) < Flatter.goal_max_drop(parent)
                end
            end
        end

        @testset "proved subgoals keep top_N, so the curve is shared" begin
            parent = Flatter.proved_goal(64, 0.5)
            child = Flatter.subgoal(parent, 0, 16)
            @test child.n == 16
            @test child.top_N == 64
            @test child.proved
            # The child's slope is the parent's curve read at the smaller
            # dimension, not a fresh interpolation.
            @test Flatter.goal_slope(child) ≈ Flatter.goal_alpha(parent, 16) rtol = 1e-12
        end

        @testset "range checking" begin
            goal = Flatter.goal_from_slope(16, 0.1)
            @test_throws ArgumentError Flatter.subgoal(goal, 4, 4)
            @test_throws ArgumentError Flatter.subgoal(goal, 0, 17)
        end
    end

    @testset "with_best_slope" begin
        # The budget is invariant: a heuristic goal's target slope splits as
        # best_slope + gap, and changing the base slope only moves the total
        # between the two terms.
        @testset "the target slope is preserved" begin
            for n in (8, 32, 64), target in (0.1, 0.2, 0.5)
                goal = Flatter.goal_from_slope(n, target)
                before = Flatter.goal_max_drop(goal)
                for new_slope in (0.001, 0.02, 0.05)
                    new_slope < target || continue
                    moved = Flatter.with_best_slope(goal, new_slope)
                    @test moved.best_slope ≈ new_slope
                    @test Flatter.goal_max_drop(moved) ≈ before rtol = 1e-10
                    @test Flatter.goal_max_drop(moved) / n ≈ target rtol = 1e-10
                end
            end
        end

        @testset "quality absorbs the change" begin
            goal = Flatter.goal_from_slope(32, 0.2)
            shape = Flatter._goal_shape(32)
            for new_slope in (0.005, 0.02)
                moved = Flatter.with_best_slope(goal, new_slope)
                # gap = quality * shape / n is the excess over the base slope;
                # it must grow by exactly the drop in base slope.
                old_gap = goal.quality * shape / 32
                new_gap = moved.quality * shape / 32
                @test new_gap - old_gap ≈ goal.best_slope - new_slope rtol = 1e-10
            end
        end

        @testset "round trips back to the original goal" begin
            goal = Flatter.goal_from_slope(16, 0.3)
            there = Flatter.with_best_slope(goal, 0.01)
            back = Flatter.with_best_slope(there, goal.best_slope)
            @test back.best_slope ≈ goal.best_slope rtol = 1e-12
            @test back.quality ≈ goal.quality rtol = 1e-10
        end

        # The precondition is slope < target slope, not merely slope small.
        @testset "rejects a base slope at or above the target" begin
            goal = Flatter.goal_from_slope(16, 0.05)
            @test_throws ArgumentError Flatter.with_best_slope(goal, 10.0)
            @test_throws ArgumentError Flatter.with_best_slope(goal, 0.05)
            # Just below the target is fine.
            @test Flatter.goal_max_drop(Flatter.with_best_slope(goal, 0.049)) ≈
                  Flatter.goal_max_drop(goal) rtol = 1e-10
        end

        @testset "does not apply to proved goals" begin
            @test_throws ArgumentError Flatter.with_best_slope(
                Flatter.proved_goal(16, 0.4), 0.05)
        end
    end

    @testset "goal_check, heuristic" begin
        @testset "dimension one is accepted unconditionally" begin
            @test Flatter.goal_check(Flatter.goal_from_slope(1, 0.1), [1e9])
        end

        @testset "a flat profile is accepted, n = $n" for n in 2:40
            @test Flatter.goal_check(Flatter.goal_from_slope(n, 0.1),
                                     goal_flat_profile(n))
        end

        @testset "a steep profile is rejected, n = $n" for n in 2:40
            goal = Flatter.goal_from_slope(n, 0.1)
            @test !Flatter.goal_check(goal, goal_linear_profile(n, 100.0, 10.0))
        end

        @testset "the profile length must match the goal" begin
            goal = Flatter.goal_from_slope(8, 0.1)
            @test_throws DimensionMismatch Flatter.goal_check(goal, goal_flat_profile(7))
            @test_throws DimensionMismatch Flatter.goal_check(goal, goal_flat_profile(9))
        end

        # This is what distinguishes the heuristic goal from the proved one. A
        # profile that is flat within each half but steps down between them has
        # total drop equal to the step, so the drop condition alone would accept
        # it while the basis is still improvable across that gap.
        @testset "a step between the halves is caught even when the total drop fits" begin
            for n in (8, 16, 32)
                goal = Flatter.goal_from_slope(n, 0.1)
                budget = Flatter.goal_max_drop(goal)
                n_L = div(n, 2)

                small_step = [fill(budget * 0.5, n_L); zeros(n - n_L)]
                large_step = [fill(budget * 0.95, n_L); zeros(n - n_L)]

                # Both have total drop below budget.
                @test Flatter.profile_drop(small_step) < budget
                @test Flatter.profile_drop(large_step) < budget

                @test Flatter.goal_check(goal, small_step)
                @test !Flatter.goal_check(goal, large_step)
            end
        end

        # Condition 3 is not implied by conditions 1 and 2: there exist profiles
        # passing both that it rejects. Without this the middle condition could
        # be dropped as redundant.
        @testset "the middle-span condition binds independently" begin
            rng = MersenneTwister(0x60A1)
            found_binding = false
            for _ in 1:20_000
                n = rand(rng, (8, 16))
                goal = Flatter.goal_from_slope(n, 0.1)
                budget = Flatter.goal_max_drop(goal)
                prof = sort(rand(rng, n) .* (budget * 1.1); rev = true)

                n_L = div(n, 2)
                n_R = n - n_L
                gamma = goal.quality * Float64(n)^log2(3.0)
                mu_sep = (budget - gamma) / 2 + gamma
                mu_L = sum(view(prof, 1:n_L)) / n_L
                mu_R = sum(view(prof, (n_L + 1):n)) / n_R

                first_two = Flatter.profile_drop(prof) < budget && mu_L - mu_R < mu_sep
                if first_two && !Flatter.goal_check(goal, prof)
                    found_binding = true
                    break
                end
            end
            @test found_binding
        end

        @testset "accepting is monotone in the goal" begin
            # A profile accepted by a strict goal is accepted by a looser one.
            rng = MersenneTwister(0x60A2)
            for _ in 1:200
                n = rand(rng, 4:24)
                prof = sort(rand(rng, n) .* 3.0; rev = true)
                strict = Flatter.goal_from_slope(n, 0.05)
                loose = Flatter.goal_from_slope(n, 0.5)
                if Flatter.goal_check(strict, prof)
                    @test Flatter.goal_check(loose, prof)
                end
            end
        end
    end

    @testset "argument checking" begin
        @test_throws ArgumentError Flatter.goal_from_slope(0, 0.1)
        @test_throws ArgumentError Flatter.goal_from_drop(1, 1.0)
        @test_throws ArgumentError Flatter.goal_check(Flatter.ReductionGoal(), Float64[])
    end
end
