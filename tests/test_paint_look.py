"""The paint post pass ([look] post = "paint"): the frame redrawn as dabs of pure palette colour that mix in the eye."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import CONFIG, make_project, sphere_text, write_model

from kiln import post
from kiln.build import build
from kiln.godot import find_godot_binary, look_scene, verify
from kiln.project import Project, load_project
from kiln.style import Look, load_style

PAINT = '\n[look]\npost = "paint"\npost_palette = "pointillist"\npost_lines = 60\n'


def _paint_project(root: Path, look: str = PAINT, extra: str = "") -> Project:
    project = make_project(root, CONFIG + extra)
    (project.root / "styles" / "fast.style.toml").write_text((project.root / "styles" / "fast.style.toml").read_text() + look)
    write_model(project, sphere_text())
    return project


def _look(colors: tuple[str, ...], **keys: object) -> Look:
    return replace(Look(post="paint", post_palette="p", post_colors=colors, post_lines=20, pixel_aspect=1.0), **keys)  # type: ignore[arg-type]


def _colours(image: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(int(v) for v in np.round(c * 255)) for c in image[..., :3].reshape(-1, 3)}


def _palette(look: Look) -> set[tuple[int, ...]]:
    return {tuple(int(round(v * 255)) for v in post.rgb(c)) for c in look.post_colors}


def test_presets() -> None:
    style, findings = load_style("paint_pointillist")
    assert style is not None and not findings
    look = style.look
    assert (look.post, look.post_palette, look.post_lines, look.dither_space) == ("paint", "pointillist", 90, "view")
    signac, findings = load_style("paint_signac")  # the environment only: its dabs are a surface shader in the retro room
    assert signac is not None and not findings
    assert signac.look.post == "" and signac.look.tonemapper == "linear"
    assert style.smooth and signac.smooth  # both extend soft_toy: gradients become dab mixtures


def test_init_keeps_the_starter_palette(tmp_path: Path) -> None:
    """A retro style's `kiln init` writes the hardware palette as `main` (models can only show those colours); a paint
    look mixes any colour from its dabs, so models keep the starter palette."""
    from kiln.cli import main

    assert main(["init", str(tmp_path / "p"), "--style", "paint_pointillist"]) == 0
    assert main(["init", str(tmp_path / "plain"), "--style", "soft_toy"]) == 0
    project = load_project(start=tmp_path / "p")
    style, findings = project.style()
    assert style is not None and not findings and len(style.look.post_colors) == 16  # the pass's palette is the library's
    assert (project.root / "palettes" / "main.pal.yaml").read_text() == (tmp_path / "plain" / "palettes" / "main.pal.yaml").read_text()
    write_model(project, sphere_text())
    model, _, mf = project.load(project.models_dir / "thing.kiln.yaml")
    assert model is not None and not mf


def test_look_scene_starts_in_paint_mode(tmp_path: Path) -> None:
    style, findings = _paint_project(tmp_path / "a").style()
    assert style is not None and not findings
    scene = look_scene(style)
    for text in ("shader_parameter/mode = 1", "shader_parameter/count = 16", "shader_parameter/dab_size = 1.4",
                 "shader_parameter/shadow_tint = Vector3(0.42, 0.36, 0.78)", "shader_parameter/ground = Vector4(0.95, 0.92, 0.84, 0.0)",
                 "func set_mode", "func set_paint", "func get_paint", "vec3 paint(vec2 screen_uv"):  # fmt: skip
        assert text in scene, text
    assert scene.count("shader_parameter/mode") == 1  # the canvas pass only: the World pass has no dabs
    world = post.world_params(style.look)
    assert [line.split(" ")[0].split("/")[1] for line in world] == [*post.WORLD_UNIFORMS, "exposure"]
    retro, _ = load_style("retro_cga")
    assert retro is not None
    # a retro kit is the same shader in mode 0, with the dab settings at their defaults for a game that switches
    lines = post.material_params(replace(retro.look, post_colors=("#000000", "#ffffff")))
    assert "shader_parameter/mode = 0" in lines and "shader_parameter/dab_size = 1.4" in lines
    assert post.shader(retro.look) == post.shader(style.look)


def test_shader_and_numpy_share_the_settings() -> None:
    for name, value in post.PAINT_DEFAULTS.items():
        assert f"uniform float {name} = {post._float(value)};" in post.PAINT_GLSL, name
    assert all(f'"{name}"' in post.RETRO_SCRIPT for name in post.PAINT_DEFAULTS)  # the Retro node's PAINT list
    assert f"uniform vec3 shadow_tint = vec3({', '.join(post._float(v) for v in post.SHADOW_TINT)});" in post.PAINT_GLSL
    assert f"uniform vec4 ground = vec4({', '.join(post._float(v) for v in post.GROUND)});" in post.PAINT_GLSL


def test_paint_only_draws_palette_colours_in_dabs() -> None:
    rng = np.random.default_rng(1)
    image = rng.random((80, 120, 3))
    look = _look(("#fff6e0", "#ff8c69", "#7d5bd6", "#5dbb63"))
    out = post.paint(image, look)
    assert out.shape == image.shape and _colours(out) <= _palette(look)
    assert (post.retro(image, look) == out).all()  # previews go through retro(): a paint look draws its dabs
    assert (post.paint(image, look) == out).all()
    blocks = out.reshape(20, 4, 30, 4, 3)  # 20 lines of 4 px: dabs are jittered and turned, not the retro mode's blocks
    assert not (blocks == blocks[:, :1, :, :1]).all()
    assert (post.retro(image, replace(look, post="retro")) != out).any()


def test_dabs_mix_in_proportion() -> None:
    """A flat colour between two palette colours comes out as dabs of both, in proportion (each dab rolls its own
    threshold); the grade is taken out by asking for the colour it gives."""
    look = _look(("#000000", "#ffffff"))
    for grey, low, high in ((0.25, 0.15, 0.4), (0.5, 0.4, 0.6), (0.75, 0.6, 0.85)):
        s = post.PAINT_DEFAULTS
        graded = float(post.painterly(np.full((1, 3), grey), s["vivid"], s["lift"], s["shadow_hue"]).mean())
        out = post.paint(np.full((200, 200, 3), grey), look)
        white = float((out[..., 0] > 0.5).mean())
        assert abs(white - graded) < 0.08 and low < graded < high + 0.2, (grey, graded, white)


def test_painterly_grade() -> None:
    s = post.PAINT_DEFAULTS

    def grade(c: list[float]) -> np.ndarray:
        graded: np.ndarray = post.painterly(np.array([c]), s["vivid"], s["lift"], s["shadow_hue"])[0]
        return graded

    black, white, red = grade([0.0, 0.0, 0.0]), grade([1.0, 1.0, 1.0]), grade([0.6, 0.3, 0.3])
    assert black[2] > black[0] > 0.2 and black[2] > black[1]  # shadows turn violet instead of going dark
    np.testing.assert_allclose(white, 1.0)
    assert red[0] - red[1] > 0.3 * (1 - s["lift"])  # more colour than went in (0.3), then lifted towards white


def test_alpha_is_the_blocks_snapped() -> None:
    image = np.zeros((40, 40, 4))
    image[:, 20:, 3] = 0.8
    image[:, :20, 3] = 0.3
    out = post.paint(image, _look(("#000000", "#ffffff")))
    assert set(np.unique(out[..., 3])) == {0.0, 1.0} and out[:, 20:, 3].all() and not out[:, :20, 3].any()


def test_previews_draw_the_dabs(tmp_path: Path) -> None:
    from kiln.builder import build_model
    from kiln.render import View, render

    project = _paint_project(tmp_path)
    model, style, findings = project.load(project.models_dir / "thing.kiln.yaml")
    assert model is not None and style is not None and not findings
    built = build_model(model, style)
    image = np.asarray(render(built, View(30.0, 15.0), 200, background=(40, 40, 40, 255)))
    assert {tuple(int(v) for v in c) for c in image[..., :3].reshape(-1, 3)} <= _palette(style.look)
    assert (np.asarray(render(built, View(30.0, 15.0), 200, background=(40, 40, 40, 255), post=False)) != image).any()


@pytest.mark.skipif(find_godot_binary() is None, reason="needs a Godot 4 binary (GODOT=/path)")
@pytest.mark.parametrize("space", ["screen", "view"])
def test_godot_loads_the_pass(tmp_path: Path, space: str) -> None:
    godot = find_godot_binary()
    assert godot is not None
    (tmp_path / "project.godot").write_text("config_version=5\n")
    project = _paint_project(tmp_path / "art", PAINT + f'dither = "noise"\ndither_space = "{space}"\n',
                             extra='output_dir = "../assets/kiln"\nexport = ["glb", "godot"]\n')
    result = build(project)
    assert not result.errors and result.godot_root is not None
    assert "shader_parameter/mode = 1" in (project.output_dir / "looks" / "fast.tscn").read_text()
    check = verify(godot, result.godot_root, "res://assets/kiln/manifest.json")
    assert check.ok, check.output[-3000:]


RUNTIME_GD = """extends SceneTree
func _init() -> void:
\tvar kit: Node = load(OS.get_cmdline_user_args()[0]).instantiate()
\troot.add_child(kit)
\tvar retro: Node = kit.get_node("Retro")
\tretro._ready()
\tprint("A ", retro.get_mode(), " ", snapped(retro.get_paint("dab_size"), 0.01), " ", retro.get_paint("ground"))
\tretro.set_mode(0)
\tretro.set_paint("dab_size", 2)
\tretro.set_paint("lift", 0.5)
\tretro.set_paint("shadow_tint", Color(0.25, 0.5, 0.75))
\tretro.set_paint("ground", Color(1.0, 0.5, 0.0, 0.5))
\tretro.set_paint("no_such_setting", 3.0)
\tprint("B ", retro.get_mode(), " ", retro.get_paint("dab_size"), " ", retro.get_paint("lift"), " ", retro.get_paint("shadow_tint"), " ", retro.get_paint("ground"))
\tquit(0)
"""


@pytest.mark.slow
@pytest.mark.skipif(find_godot_binary() is None, reason="needs a Godot 4 binary (GODOT=/path)")
def test_godot_tunes_the_dabs_at_runtime(tmp_path: Path) -> None:
    import os
    import subprocess

    godot = find_godot_binary()
    assert godot is not None
    (tmp_path / "project.godot").write_text("config_version=5\n")
    project = _paint_project(tmp_path / "art", extra='output_dir = "../assets/kiln"\nexport = ["glb", "godot"]\n')
    assert not build(project).errors
    (tmp_path / "runtime.gd").write_text(RUNTIME_GD)
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    subprocess.run([godot, "--headless", "--path", str(tmp_path), "--import"], capture_output=True, timeout=300, env=env)
    run = subprocess.run([godot, "--headless", "--path", str(tmp_path), "-s", "runtime.gd", "--", "res://assets/kiln/looks/fast.tscn"],
                         capture_output=True, text=True, timeout=300, env=env)  # fmt: skip
    lines = [line for line in run.stdout.splitlines() if line[:2] in ("A ", "B ")]
    assert lines == ["A 1 1.4 (0.95, 0.92, 0.84, 0.0)", "B 0 2.0 0.5 (0.25, 0.5, 0.75, 1.0) (1.0, 0.5, 0.0, 0.5)"], run.stdout + run.stderr
