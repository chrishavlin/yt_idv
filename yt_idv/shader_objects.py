"""
Shader and ShaderProgram wrapper classes for vertex and fragment shaders used
in Interactive Data Visualization
"""

import contextlib
import ctypes
import os
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import traitlets
import yaml
from OpenGL import GL
from yt.units.yt_array import YTQuantity
from yt.utilities.exceptions import (
    YTInvalidShaderType,
    YTUnknownUniformKind,
    YTUnknownUniformSize,
)

from .opengl_support import GLValue, num_to_const

_NULL_SOURCES = {
    "geometry": r"""
#version 330 core
layout ( points ) in;
layout ( points ) out;

void main() {
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
}
    """,
    "vertex": r"""
#version 330 core

// Input vertex data, different for all executions of this shader.
in vec3 vertexPosition_modelspace;

// Output data ; will be interpolated for each fragment.
out vec2 UV;

void main()
{
    gl_Position = vec4(vertexPosition_modelspace, 1.0);
    UV = (vertexPosition_modelspace.xy+vec2(1.0,1.0))/2.0;
}

""",
    "fragment": r"""
#version 330 core

out vec4 color;

void main() {
    color = vec4(gl_FragCoord.x, gl_FragCoord.y, gl_FragCoord.z, 1.0);
    return;
}
""",
}


class ShaderProgram:
    """
    Wrapper class that compiles and links vertex and fragment shaders
    into a shader program.

    Parameters
    ----------

    vertex_shader : string
        or :class:`yt.visualization.volume_rendering.shader_objects.VertexShader`
        The vertex shader used in the Interactive Data Visualization pipeline.

    fragment_shader : string
        or :class:`yt.visualization.volume_rendering.shader_objects.FragmentShader`
        The fragment shader used in the Interactive Data Visualization pipeline.

    geometry_shader : string
        or :class:`yt_idv.shader_objects.GeometryShader`
        The geometry shader used in the pipeline; optional.

    preprocessor_defs : PreprocessorDefinitionState
        a PreprocessorDefinitionState instance defining any preprocessor
        definitions if used; optional.

    """

    def __init__(
        self,
        vertex_shader=None,
        fragment_shader=None,
        geometry_shader=None,
        preprocessor_defs=None,
    ):
        # Don't allow just one.  Either neither or both.
        if vertex_shader is None and fragment_shader is None:
            pass
        elif None not in (vertex_shader, fragment_shader):
            # Geometry is optional
            self.link(
                vertex_shader,
                fragment_shader,
                geometry_shader,
                preprocessor_defs=preprocessor_defs,
            )
        else:
            raise RuntimeError
        self._uniform_funcs = OrderedDict()

    def link(
        self,
        vertex_shader,
        fragment_shader,
        geometry_shader=None,
        preprocessor_defs=None,
    ):
        if preprocessor_defs is None:
            preprocessor_defs = PreprocessorDefinitionState()

        # We allow an optional geometry shader, but not tesselation (yet?)
        self.program = GL.glCreateProgram()
        if not isinstance(vertex_shader, Shader):
            vertex_shader = Shader(
                source=vertex_shader,
                preprocessor_defs=preprocessor_defs.get_shader_defs("vertex"),
            )
        if not isinstance(fragment_shader, Shader):
            fragment_shader = Shader(
                source=fragment_shader,
                preprocessor_defs=preprocessor_defs.get_shader_defs("fragment"),
            )
        if geometry_shader is not None and not isinstance(geometry_shader, Shader):
            geometry_shader = Shader(
                source=geometry_shader,
                preprocessor_defs=preprocessor_defs.get_shader_defs("geometry"),
            )
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        self.geometry_shader = geometry_shader
        GL.glAttachShader(self.program, vertex_shader.shader)
        GL.glAttachShader(self.program, fragment_shader.shader)
        if geometry_shader is not None:
            GL.glAttachShader(self.program, geometry_shader.shader)
        GL.glLinkProgram(self.program)
        result = GL.glGetProgramiv(self.program, GL.GL_LINK_STATUS)
        if not result:
            raise RuntimeError(GL.glGetProgramInfoLog(self.program))
        vertex_shader.delete_shader()
        fragment_shader.delete_shader()
        if geometry_shader is not None:
            geometry_shader.delete_shader()
        self.introspect()

    def introspect(self):
        if self.program is None:
            raise RuntimeError
        # First get all of the uniforms
        self.uniforms = {}
        self.attributes = {}

        if not bool(GL.glGetProgramInterfaceiv):
            return
        n_uniforms = GL.glGetProgramInterfaceiv(
            self.program, GL.GL_UNIFORM, GL.GL_ACTIVE_RESOURCES
        )

        for i in range(n_uniforms):
            name, size, gl_type = GL.glGetActiveUniform(self.program, i)
            if isinstance(name, np.ndarray):
                # fix for https://github.com/yt-project/yt_idv/issues/202
                # until an upstream fix is in.
                name = name.tobytes().rstrip(b"\000")
            gl_type = num_to_const[gl_type]
            self.uniforms[name.decode("utf-8")] = (size, gl_type)

        n_attrib = GL.glGetProgramInterfaceiv(
            self.program, GL.GL_PROGRAM_INPUT, GL.GL_ACTIVE_RESOURCES
        )
        length = ctypes.pointer(ctypes.c_int())
        size = ctypes.pointer(ctypes.c_int())
        gl_type = ctypes.pointer(ctypes.c_int())
        name = ctypes.create_string_buffer(256)
        for i in range(n_attrib):
            GL.glGetActiveAttrib(self.program, i, 256, length, size, gl_type, name)
            gl_const = num_to_const[gl_type[0]]
            self.attributes[name[: length[0]].decode("utf-8")] = (size[0], gl_const)

    def delete_program(self):
        if self.program is not None:
            GL.glDeleteProgram(self.program)
            self.program = None

    def _guess_uniform_func(self, value):
        # We make a best-effort guess.
        # This does NOT work with arrays of uniforms.
        # First look at the dtype kind.  Fortunately, this falls into either
        # 'f' or 'i', which matches nicely with OpenGL.
        # Note that in some implementations, it seems there is also a 'd' type,
        # but we will not be using that here.
        if isinstance(value, int):
            return GL.glUniform1i
        elif isinstance(value, (YTQuantity, float)):
            return GL.glUniform1f
        else:
            kind = value.dtype.kind
        if kind not in "if":
            raise YTUnknownUniformKind(kind)
        if value.ndim == 0:
            return {"f": GL.glUniform1f, "i": GL.glUniform1i}[kind]
        elif value.ndim == 1:
            # This is not precisely the breakdown, but it lets us
            # pass in arrays that are greater than 4 long as arrays
            if value.size > 4:
                func = self._set_array_uniform(kind, value.size)
            else:
                func = self._set_scalar_uniform(kind, value.size)
        elif value.ndim == 2:
            if value.shape[0] != value.shape[1]:
                raise YTUnknownUniformSize(value.shape)
            func = self._set_matrix_uniform(kind, value.shape)
        else:
            raise YTUnknownUniformSize(value.shape)
        return func

    def _set_array_uniform(self, kind, size_spec):
        gl_func = getattr(GL, f"glUniform1{kind}v")

        def _func(location, value):
            return gl_func(location, size_spec, value)

        return _func

    def _set_scalar_uniform(self, kind, size_spec):
        gl_func = getattr(GL, f"glUniform{size_spec}{kind}v")

        def _func(location, value):
            return gl_func(location, 1, value)

        return _func

    def _set_matrix_uniform(self, kind, size_spec):
        assert size_spec[0] == size_spec[1]
        gl_func = getattr(GL, f"glUniformMatrix{size_spec[0]}{kind}v")

        def _func(location, value):
            return gl_func(location, 1, GL.GL_TRUE, value)

        return _func

    def _set_uniform(self, name, value):
        # We need to figure out how to pass it in.
        if name not in self._uniform_funcs:
            self._uniform_funcs[name] = self._guess_uniform_func(value)
        loc = GL.glGetUniformLocation(self.program, name)
        if loc < 0:
            return -1
        return self._uniform_funcs[name](loc, value)

    @contextlib.contextmanager
    def enable(self):
        GL.glUseProgram(self.program)
        self.vertex_shader.setup_blend()
        self.fragment_shader.setup_blend()
        yield self
        GL.glUseProgram(0)


class Shader(traitlets.HasTraits):
    """
    Creates a shader from source

    Parameters
    ----------

    source : str
        This can either be a string containing a full source of a shader,
        an absolute path to a source file or a filename of a shader
        residing in the ./shaders/ directory.
    allow_null : bool
        If True (default) then shader compilation errors will be caught and
        printed without raising exception (convenient for general use and for
        developing new shaders). If False, any compilation errors will raise
        a RunTimeError (useful for CI testing).

    """

    _shader = None
    allow_null = traitlets.Bool(True)
    source = traitlets.Any()
    shader_name = traitlets.CUnicode()
    info = traitlets.CUnicode()
    shader_type = traitlets.CaselessStrEnum(("vertex", "fragment", "geometry"))
    blend_func = traitlets.Tuple(
        GLValue(), GLValue(), default_value=("src alpha", "dst alpha")
    )
    blend_equation = GLValue("func add")
    depth_test = GLValue("always")
    _filename = traitlets.Unicode()

    use_separate_blend = traitlets.Bool(False)
    blend_equation_separate = traitlets.Tuple(
        GLValue(), GLValue(), default_value=("none", "none")
    )
    blend_func_separate = traitlets.Tuple(
        GLValue(),
        GLValue(),
        GLValue(),
        GLValue(),
        default_value=("none", "none", "none", "none"),
    )
    preprocessor_defs = traitlets.List(trait=traitlets.Tuple())

    def _get_source(self, source):
        if ";" in source:
            # This is probably safe, right?  Enh, probably.
            return source

        # What this does is concatenate multiple (if available) source files.
        # This gets around GLSL's composition issues, which means we can have
        # functions that get called at each step in a ray tracing process, for
        # instance, that can still share ray tracing code between multiple
        # files.
        if not isinstance(source, (tuple, list)):
            source = (source,)
        source = (
            ("header.inc.glsl",)
            + tuple(self.preprocessor_defs)
            + ("known_uniforms.inc.glsl",)
            + tuple(source)
        )
        full_source = []
        for fn in source:
            if isinstance(fn, tuple):
                full_source.append(f"#define {fn[0]} {fn[1]}\n")
                continue
            if os.path.isfile(fn):
                sh_directory = ""
            else:
                sh_directory = os.path.join(os.path.dirname(__file__), "shaders")
            fn = os.path.join(sh_directory, fn)
            if not os.path.isfile(fn):
                raise YTInvalidShaderType(fn)
            self._filename = fn
            full_source.append(open(fn).read())
        return "\n\n".join(full_source)

    def _enable_null_shader(self):
        source = _NULL_SOURCES[self.shader_type]
        self.compile(source=source)

    @property
    def defines(self):
        return "\n".join([f"#define {var}" for var in self.preprocessor_defs])

    def compile(self, source=None):
        if source is None:
            source = self.source
            if source is None:
                raise RuntimeError
        source = self._get_source(source)
        shader_type_enum = getattr(GL, f"GL_{self.shader_type.upper()}_SHADER")
        shader = GL.glCreateShader(shader_type_enum)
        # We could do templating here if we wanted.
        self.shader_source = source
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        result = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        if not (result):
            msg = f"shader complilation error for {self.shader_name} {self.shader_type}"
            if self._filename is not None:
                msg += f" in {self._filename}."
            msg += f"\n GL.glGetShaderInfoLog: \n {GL.glGetShaderInfoLog(shader)}"
            raise RuntimeError(msg)
        self._shader = shader

    def setup_blend(self):
        GL.glEnable(GL.GL_BLEND)
        if self.use_separate_blend:
            GL.glBlendEquationSeparate(*self.blend_equation_separate)
            GL.glBlendFuncSeparate(*self.blend_func_separate)
        else:
            GL.glBlendEquation(self.blend_equation)
            GL.glBlendFunc(*self.blend_func)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(self.depth_test)

    @property
    def shader(self):
        if self._shader is None:
            try:
                self.compile()
            except RuntimeError as exc:
                if self.allow_null:
                    print(exc)
                    for line_num, line in enumerate(self.shader_source.split("\n")):
                        print(f"{line_num + 1:05}: {line}")
                    self._enable_null_shader()
                else:
                    raise exc
        return self._shader

    def delete_shader(self):
        if None not in (self._shader, GL.glDeleteShader):
            GL.glDeleteShader(self._shader)
            self._shader = None

    def __del__(self):
        # This is not guaranteed to be called
        self.delete_shader()


def _validate_shader(shader_type, value, allow_null=True, preprocessor_defs=None):
    shader_info = known_shaders[shader_type][value]
    shader_info.setdefault("shader_type", shader_type)
    shader_info["use_separate_blend"] = bool("blend_func_separate" in shader_info)
    shader_info.setdefault("shader_name", value)
    if preprocessor_defs is not None:
        shader_info["preprocessor_defs"] = preprocessor_defs
    return Shader(allow_null=allow_null, **shader_info)


class ShaderTrait(traitlets.TraitType):
    default_value = None
    info_text = "A shader (vertex, fragment or geometry)"

    def validate(self, obj, value):
        if isinstance(value, str) or isinstance(value, tuple):
            try:
                shader_type = self.metadata.get("shader_type", "vertex")
                if isinstance(value, tuple):
                    preprocessor_defs = value[1]
                    value = value[0]
                else:
                    preprocessor_defs = None
                return _validate_shader(
                    shader_type, value, preprocessor_defs=preprocessor_defs
                )
            except KeyError:
                self.error(obj, value)
        elif isinstance(value, Shader):
            return value
        self.error(obj, value)


class PreprocessorDefinitionState:

    _valid_shader_types = ("vertex", "geometry", "fragment")

    def __init__(self):
        self.vertex = {}
        self.geometry = {}
        self.fragment = {}

    def _get_dict(self, shader_type: str) -> dict:
        """return the dict of definitions for specifed shader_type"""
        return getattr(self, shader_type)

    def add_definition(self, shader_type: str, value: Tuple[str, str]):
        """add a definition for specified shader_type, will overwrite
        existing definitions.
        """
        self._validate_shader_type(shader_type)
        self._get_dict(shader_type)[value[0]] = value[1]

    def clear_definition(self, shader_type: str, value: Tuple[str, str]):
        """remove the definition of value for specified shader_type"""
        self._validate_shader_type(shader_type)
        self._get_dict(shader_type).pop(value[0])

    def get_shader_defs(self, shader_type: str) -> List[Tuple[str, str]]:
        """return the preprocessor definition list for specified shader_type"""
        self._validate_shader_type(shader_type)
        return list(self._get_dict(shader_type).items())

    def _validate_shader_type(self, shader_type: str):
        if shader_type not in self._valid_shader_types:
            raise ValueError(
                f"shader_type must be one of {self._valid_shader_types}, "
                f"but found {shader_type}"
            )

    def __getitem__(self, item: str) -> List[Tuple[str, str]]:
        return self.get_shader_defs(item)

    def reset(self, shader_type: Optional[str] = None):
        if shader_type is None:
            self.vertex = {}
            self.geometry = {}
            self.fragment = {}
        else:
            self._validate_shader_type(shader_type)
            setattr(self, shader_type, {})


known_shaders = {}
component_shaders = {}
default_shader_combos = {}

# We'll load our shaders here from shaderlist.yaml
_shlist_fn = os.path.join(os.path.dirname(__file__), "shaders", "shaderlist.yaml")
if os.path.exists(_shlist_fn):
    with open(_shlist_fn) as f:
        shader_info = yaml.load(f, yaml.SafeLoader)
    known_shaders.update(shader_info["shader_definitions"])
    component_shaders.update(shader_info["component_shaders"])
    default_shader_combos.update(
        {_: component_shaders[_].pop("default_value") for _ in component_shaders}
    )


def get_shader_combos(component_name, coord_system="cartesian"):
    shader_combos = list(sorted(component_shaders[component_name]))
    if coord_system == "cartesian":
        return shader_combos

    valid_shader_combos = []
    for shader_name in shader_combos:
        shader = component_shaders[component_name][shader_name]
        if (
            "coordinate_systems" in shader
            and coord_system in shader["coordinate_systems"]
        ):
            valid_shader_combos.append(shader_name)
    return valid_shader_combos
