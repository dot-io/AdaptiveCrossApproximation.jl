# Generate a figure of the benchmark geometry: two well-separated unit
# spherical meshes, using exactly the same `meshsphere` + `translate` calls
# as the throughput benchmarks. Outputs both a vector PDF (for slides /
# the thesis) and a PNG (for quick inspection).
#
# Run: julia --project=. bench/plot_test_setup.jl
#
# Dependencies (add once into your env):
#   using Pkg; Pkg.add(["CairoMakie", "GeometryBasics",
#                       "CompScienceMeshes", "StaticArrays"])

using CompScienceMeshes
using StaticArrays
using CairoMakie
using GeometryBasics: Point3f, TriangleFace
import GeometryBasics

# --- Geometry: matches bench/bench_hmatrix.jl ---------------------------
const H          = 0.30          # mesh edge-length parameter
const SEPARATION = 3.0           # centre-to-centre distance along x

Γ1 = CompScienceMeshes.translate(
        meshsphere(1.0, H), SVector(-SEPARATION / 2, 0.0, 0.0))
Γ2 = CompScienceMeshes.translate(
        meshsphere(1.0, H), SVector( SEPARATION / 2, 0.0, 0.0))

# --- Adapter: CompScienceMeshes.Mesh -> Makie ---------------------------
to_makie(m) = GeometryBasics.Mesh(
    [Point3f(v...)              for v in m.vertices],
    [TriangleFace(c.indices...) for c in m.faces],
)

# --- Figure -------------------------------------------------------------
fig = Figure(size = (1400, 700), backgroundcolor = :transparent)
ax  = Axis3(fig[1, 1];
            aspect   = :data,
            viewmode = :fit,
            elevation = 0.25,
            azimuth   = -0.6)

for (Γ, surface_colour) in ((Γ1, :steelblue), (Γ2, :tomato))
    gmesh = to_makie(Γ)
    mesh!(ax, gmesh;
          color = surface_colour, transparency = true, alpha = 0.35)
    wireframe!(ax, gmesh; color = :black, linewidth = 0.4)
end

hidedecorations!(ax)
hidespines!(ax)

# --- Output -------------------------------------------------------------
outdir = joinpath(@__DIR__, "results")
isdir(outdir) || mkpath(outdir)
save(joinpath(outdir, "test_setup.pdf"), fig)
save(joinpath(outdir, "test_setup.png"), fig; px_per_unit = 3)

println("Wrote:")
println("  ", joinpath(outdir, "test_setup.pdf"))
println("  ", joinpath(outdir, "test_setup.png"))
println("  geometry: 2× meshsphere(1.0, $H) translated by ±$(SEPARATION/2) along x")
println("  elements per sphere: ", length(Γ1.faces))
