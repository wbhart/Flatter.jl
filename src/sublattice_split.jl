# sublattice_split.jl
#
# The schedules: which windows get reduced, in what order, at each phase.
#
# This package has so far used `_reduction_window`, a hand-rolled cycle of
# middle, left and right windows taken from `Proved3`. flatter's real schedules
# are trees, built once and walked, and they differ from that approximation in a
# way that matters:
#
#   * a phase 3 node yields TWO windows on odd iterations, not one. Our version
#     only ever produced a single window per iteration, which is why the tiled
#     update had so little structure to work with -- one window means one reduced
#     tile and two gaps, where flatter routinely has two reduced tiles and none.
#   * a phase 2 node has its own three-step schedule -- left half, right half,
#     then the whole -- and hands the whole-window step to a phase 3 tree.
#
# The trees SHARE nodes on purpose. A phase 3 node's middle child is built from
# its left child's right child and its right child's left child, so those nodes
# are reachable by more than one path and their iteration state is shared. That
# is deliberate in flatter -- its destructor detaches the middle child's pointers
# before deleting to avoid a double free -- and it is reproduced here by holding
# the same objects rather than copies.
#
# Indices here are 1-based inclusive ranges; flatter's are 0-based half-open.

"""
    SplitPhase3

A node of the phase 3 schedule: alternately the middle window, then the left and
right halves.

Even iterations yield one window, the middle, spanning the inner quarters either
side of the split point, and are a stopping point. Odd iterations yield two, the
left and right halves, and are not.
"""
mutable struct SplitPhase3
    n::Int
    k::Int
    left::Union{Nothing, SplitPhase3}
    right::Union{Nothing, SplitPhase3}
    mid::Union{Nothing, SplitPhase3}
    iter::Int
end

function SplitPhase3(n::Integer, k::Integer = div(n, 2))
    n = Int(n)
    n >= 1 || throw(ArgumentError("a split needs at least one column, got $n"))
    node = SplitPhase3(n, Int(k), nothing, nothing, nothing, 0)

    if n == 3
        node.k = 2
        node.left = SplitPhase3(2)
        node.right = SplitPhase3(1)
        node.mid = SplitPhase3(node.left.right, node.right)
    elseif n >= 2
        node.left = SplitPhase3(Int(k))
        node.right = SplitPhase3(n - Int(k))
        node.mid = SplitPhase3(node.left.right, node.right.left)
    end
    return node
end

"""
    SplitPhase3(l, r)

A node spanning two existing subtrees, which it does NOT copy.

Used to build a middle child from the inner quarters of its siblings, so those
nodes belong to two parents at once and advance together.
"""
function SplitPhase3(l::Union{Nothing, SplitPhase3}, r::Union{Nothing, SplitPhase3})
    k = l === nothing ? 1 : l.n
    n = k + (r === nothing ? 1 : r.n)
    node = SplitPhase3(n, k, l, r, nothing, 0)

    if n == 3
        node.mid = l !== nothing && l.n == 2 ?
                   SplitPhase3(l.right, r) : SplitPhase3(l, r === nothing ? nothing : r.left)
    elseif l !== nothing && l.right !== nothing && r !== nothing && r.left !== nothing
        node.mid = SplitPhase3(l.right, r.left)
    end
    return node
end

"""
    split_windows(node) -> Vector{UnitRange{Int}}

The windows this node wants reduced at its current iteration.

One window on even iterations, two on odd. Two is the case our previous
hand-rolled schedule never produced.
"""
function split_windows(node::SplitPhase3)
    if node.n == 3
        return iseven(node.iter) ? [1:2] : [2:3]
    end
    if iseven(node.iter)
        # The middle window: the inner quarter of each half.
        return [(node.left.k + 1):(node.k + node.right.k)]
    end
    return [1:node.k, (node.k + 1):node.n]
end

"""
    split_children(node) -> Vector

The child schedules matching [`split_windows`](@ref), one per window and in the
same order.
"""
function split_children(node::SplitPhase3)
    if node.n == 3
        if node.k == 2
            return iseven(node.iter) ? [node.left] : [node.mid]
        end
        return iseven(node.iter) ? [node.mid] : [node.right]
    end
    return iseven(node.iter) ? [node.mid] : [node.left, node.right]
end

"A stopping point is an iteration after which the schedule may be halted."
split_stopping_point(node::SplitPhase3) = iseven(node.iter)

split_advance!(node::SplitPhase3) = (node.iter += 1; node)

"""
    SplitPhase2

A node of the phase 2 schedule: left half, right half, then the whole.

The whole-window step hands off to a phase 3 tree, which is where flatter moves
from `Heuristic2` to `Heuristic3`. A node of three columns skips the right-half
step, giving a two-step schedule.
"""
mutable struct SplitPhase2
    n::Int
    k::Int
    left::Union{Nothing, SplitPhase2}
    right::Union{Nothing, SplitPhase2}
    all::SplitPhase3
    iter::Int
end

"`n/2`, except at three columns where the left half is two."
split_next_smaller(n::Integer) = n == 3 ? 2 : div(Int(n), 2)

function SplitPhase2(n::Integer)
    n = Int(n)
    n >= 1 || throw(ArgumentError("a split needs at least one column, got $n"))
    if n == 1
        return SplitPhase2(1, 1, nothing, nothing, SplitPhase3(1), 0)
    end
    k = split_next_smaller(n)
    left = SplitPhase2(k)
    right = SplitPhase2(n - k)
    # Shares the children's phase 3 trees, as flatter does.
    return SplitPhase2(n, k, left, right, SplitPhase3(left.all, right.all), 0)
end

function split_windows(node::SplitPhase2)
    if node.n == 3
        return node.iter == 0 ? [1:node.k] : [1:node.n]
    end
    node.iter == 0 && return [1:node.k]
    node.iter == 1 && return [(node.k + 1):node.n]
    return [1:node.n]
end

function split_children(node::SplitPhase2)
    if node.n == 3
        return node.iter == 0 ? [node.left] : [node.all]
    end
    node.iter == 0 && return [node.left]
    node.iter == 1 && return [node.right]
    return [node.all]
end

split_stopping_point(node::SplitPhase2) = node.n == 3 ? node.iter > 1 : node.iter > 2

split_advance!(node::SplitPhase2) = (node.iter += 1; node)
