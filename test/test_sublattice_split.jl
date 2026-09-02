# test/test_sublattice_split.jl
#
# Assumes `using Test` from runtests.jl.

@testset "sublattice splits" begin

    "Every window must lie inside 1:n and be non-empty."
    function ss_windows_valid(node, n)
        for w in Flatter.split_windows(node)
            (isempty(w) || first(w) < 1 || last(w) > n) && return false
        end
        return true
    end

    @testset "phase 3" begin
        @testset "alternates one middle window with two halves" begin
            node = Flatter.SplitPhase3(8)

            # Even: the middle window, spanning the inner quarter of each half.
            @test Flatter.split_windows(node) == [3:6]
            @test Flatter.split_children(node) == [node.mid]
            @test Flatter.split_stopping_point(node)

            Flatter.split_advance!(node)

            # Odd: both halves.
            @test Flatter.split_windows(node) == [1:4, 5:8]
            @test Flatter.split_children(node) == [node.left, node.right]
            @test !Flatter.split_stopping_point(node)

            Flatter.split_advance!(node)
            @test Flatter.split_windows(node) == [3:6]
        end

        @testset "two windows on odd iterations cover 1:n exactly" begin
            for n in (4, 6, 8, 16, 31, 64)
                node = Flatter.SplitPhase3(n)
                Flatter.split_advance!(node)          # to an odd iteration
                windows = Flatter.split_windows(node)
                @test length(windows) == 2
                @test reduce(vcat, [collect(w) for w in windows]) == collect(1:n)
            end
        end

        @testset "windows stay inside the node, n = $n" for n in (2, 3, 4, 5, 7, 9, 16, 33)
            node = Flatter.SplitPhase3(n)
            for _ in 1:6
                @test ss_windows_valid(node, n)
                @test length(Flatter.split_windows(node)) ==
                      length(Flatter.split_children(node))
                Flatter.split_advance!(node)
            end
        end

        @testset "a child spans exactly its window" begin
            # flatter asserts this; getting it wrong hands a sub-problem the
            # wrong number of columns.
            for n in (4, 6, 8, 12, 16, 32)
                node = Flatter.SplitPhase3(n)
                for _ in 1:4
                    for (w, c) in zip(Flatter.split_windows(node),
                                      Flatter.split_children(node))
                        c === nothing && continue
                        @test c.n == length(w)
                    end
                    Flatter.split_advance!(node)
                end
            end
        end

        @testset "the middle child is shared with its siblings" begin
            # Built from the left child's right child and the right child's left
            # child, so those nodes have two parents and advance together. The
            # sharing is deliberate in flatter.
            node = Flatter.SplitPhase3(8)
            @test node.mid.left === node.left.right
            @test node.mid.right === node.right.left

            before = node.left.right.iter
            Flatter.split_advance!(node.mid.left)
            @test node.left.right.iter == before + 1
        end

        @testset "advancing phase 3 resets the half trees" begin
            node = Flatter.SplitPhase3(8)

            # Exercise state at several depths, including a subtree shared with
            # the middle schedule.  flatter resets left and right recursively
            # before incrementing the parent, but does not reset mid.iter.
            Flatter.split_advance!(node.left)
            Flatter.split_advance!(node.left.left)
            Flatter.split_advance!(node.right)
            node.mid.iter = 7

            @test node.left.iter != 0
            @test node.left.left.iter != 0
            @test node.right.iter != 0

            Flatter.split_advance!(node)

            @test node.iter == 1
            @test node.left.iter == 0
            @test node.left.left.iter == 0
            @test node.right.iter == 0
            @test node.mid.iter == 7
        end
    end

    @testset "phase 2" begin
        @testset "runs left, right, then the whole" begin
            node = Flatter.SplitPhase2(8)

            @test Flatter.split_windows(node) == [1:4]
            @test Flatter.split_children(node) == [node.left]
            @test !Flatter.split_stopping_point(node)

            Flatter.split_advance!(node)
            @test Flatter.split_windows(node) == [5:8]
            @test Flatter.split_children(node) == [node.right]
            @test !Flatter.split_stopping_point(node)

            Flatter.split_advance!(node)
            @test Flatter.split_windows(node) == [1:8]
            # The whole-window step hands off to phase 3.
            @test Flatter.split_children(node) == [node.all]
            @test node.all isa Flatter.SplitPhase3
            @test !Flatter.split_stopping_point(node)

            Flatter.split_advance!(node)
            @test Flatter.split_stopping_point(node)
        end

        @testset "three columns skip the right half" begin
            node = Flatter.SplitPhase2(3)
            @test node.k == 2
            @test Flatter.split_windows(node) == [1:2]
            Flatter.split_advance!(node)
            @test Flatter.split_windows(node) == [1:3]
            @test Flatter.split_children(node) == [node.all]
            Flatter.split_advance!(node)
            @test Flatter.split_stopping_point(node)
        end

        @testset "the phase 3 tree is shared with the children" begin
            node = Flatter.SplitPhase2(8)
            @test node.all.left === node.left.all
            @test node.all.right === node.right.all
        end

        @testset "a child spans exactly its window, n = $n" for n in (2, 3, 4, 7, 8, 16, 33)
            node = Flatter.SplitPhase2(n)
            for _ in 1:4
                for (w, c) in zip(Flatter.split_windows(node),
                                  Flatter.split_children(node))
                    @test c.n == length(w)
                end
                @test ss_windows_valid(node, n)
                Flatter.split_advance!(node)
            end
        end
    end

    @testset "argument checking" begin
        @test_throws ArgumentError Flatter.SplitPhase3(0)
        @test_throws ArgumentError Flatter.SplitPhase2(0)
        @test_throws ArgumentError Flatter.SplitPhase2(-1)
    end
end
