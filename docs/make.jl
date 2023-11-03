using JuReto
using Documenter

DocMeta.setdocmeta!(JuReto, :DocTestSetup, :(using JuReto); recursive=true)

makedocs(;
    modules=[JuReto],
    authors="takuma I <baru.poke.15@gmail.com> and contributors",
    repo="https://github.com/takuma-i-bb/JuReto.jl/blob/{commit}{path}#{line}",
    sitename="JuReto.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://takuma-i-bb.github.io/JuReto.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/takuma-i-bb/JuReto.jl",
    devbranch="main",
)
