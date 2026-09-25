"""Tests for catanbot.vision.llm (no network: a fake client is injected)."""
from __future__ import annotations

import base64
import io
import sys
import types
from types import SimpleNamespace

import pytest
from PIL import Image

from catanbot import board as B
from catanbot.state import PHASE_MAIN
from catanbot.vision import llm
from catanbot.vision import schema as S
from catanbot.vision.result import ParseResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
HEXES = [{"resource": B.RESOURCE_NAMES[r], "number": (n if r != B.DESERT else None)} for r, n in B.STANDARD_HEXES]


def hex_ref(h: int, k: int, key: str) -> dict:
    return {"hex": h, key: k}


def make_tool_input() -> dict:
    """Canned model answer using hex-relative piece encodings."""
    return {
        "hexes": HEXES,
        "robber": 9,
        "ports": [{"hex": 0, "side": 5, "type": "3:1"}, {"hex": 2, "side": 0, "type": "wood"}],
        "players": [
            {"color": "red", "name": "Alice", "vp": 3, "cards": 5, "dev_cards": 1, "knights": 0,
             "longest_road": False, "largest_army": False,
             "settlements": [hex_ref(9, 0, "corner"), hex_ref(0, 3, "corner")],
             "cities": [hex_ref(16, 4, "corner")],
             "roads": [hex_ref(0, 5, "side"), hex_ref(9, 0, "side"), hex_ref(16, 4, "side")],
             "resources": {"wood": 2, "brick": 1, "sheep": 0, "wheat": 1, "ore": 1}},
            {"color": "blue", "name": "Bob", "vp": 2, "cards": 4, "dev_cards": 0, "knights": 1,
             "longest_road": False, "largest_army": False,
             "settlements": [hex_ref(2, 1, "corner"), hex_ref(11, 2, "corner")], "cities": [],
             "roads": [hex_ref(2, 1, "side"), hex_ref(11, 2, "side")]},
        ],
        "me": "red",
        "current_player": "blue",
        "dice": 8,
        "confidence": {"hexes": 0.95, "numbers": 0.9, "robber": 1.0, "ports": 0.6, "buildings": 0.8,
                       "roads": 0.7, "players": 0.9, "hand": 0.85, "overall": 0.8},
    }


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.messages = FakeMessages(response, error)


def tool_use_response(tool_input: dict, name: str = llm.TOOL_NAME, stop_reason: str = "tool_use"):
    """Mimic the SDK's Message object: .content list of blocks with .type/.name/.input."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Reading the board."),
            SimpleNamespace(type="tool_use", id="toolu_01", name=name, input=tool_input),
        ],
        stop_reason=stop_reason,
        stop_details=None,
        model="claude-opus-5",
        usage=SimpleNamespace(input_tokens=1234, output_tokens=567),
    )


def sample_image(size=(640, 400)) -> Image.Image:
    img = Image.new("RGB", size, (30, 90, 160))
    for x in range(0, size[0], 40):
        for y in range(0, size[1], 40):
            if (x // 40 + y // 40) % 2:
                img.paste((220, 200, 120), (x, y, x + 20, y + 20))
    return img


# ---------------------------------------------------------------------------
# conversion helpers
# ---------------------------------------------------------------------------
def test_corner_and_side_helpers_match_board_tables():
    assert llm.corner_to_vertex(9, 0) == B.HEX_VERTICES[9][0]
    assert llm.side_to_edge(0, 5) == B.HEX_EDGES[0][5]
    for h in range(B.NUM_HEXES):
        for k in range(6):
            v = llm.corner_to_vertex(h, k)
            e = llm.side_to_edge(h, k)
            # side k joins corner k and corner k+1
            assert set(B.EDGE_VERTICES[e]) == {v, llm.corner_to_vertex(h, (k + 1) % 6)}


def test_convert_pieces_maps_hex_relative_refs_to_ids():
    warnings: list = []
    parsed = llm.convert_pieces(make_tool_input(), warnings)
    assert warnings == []
    red, blue = parsed["players"]
    assert red["settlements"] == [B.HEX_VERTICES[9][0], B.HEX_VERTICES[0][3]]
    assert red["cities"] == [B.HEX_VERTICES[16][4]]
    assert red["roads"] == [B.HEX_EDGES[0][5], B.HEX_EDGES[9][0], B.HEX_EDGES[16][4]]
    assert blue["settlements"] == [B.HEX_VERTICES[2][1], B.HEX_VERTICES[11][2]]
    assert parsed["ports"] == [{"edge": B.HEX_EDGES[0][5], "type": "3:1"},
                               {"edge": B.HEX_EDGES[2][0], "type": "wood"}]
    # non-piece fields pass through untouched, including confidence
    assert parsed["hexes"] == HEXES
    assert parsed["me"] == "red" and parsed["dice"] == 8
    assert parsed["confidence"]["robber"] == 1.0
    # ports on the top-left side of hex 0 / top of hex 2 are coastal
    assert all(p["edge"] in B.COASTAL_EDGES for p in parsed["ports"])


def test_convert_pieces_merges_shared_corners_and_edges():
    # The top corner of hex 9 is also corner 2 of hex 4 and corner 4 of hex 5.
    v = B.HEX_VERTICES[9][0]
    h4 = B.HEX_VERTICES[4].index(v)
    h5 = B.HEX_VERTICES[5].index(v)
    e = B.HEX_EDGES[9][0]
    other_hex = next(h for h in B.EDGE_HEXES[e] if h != 9)
    k_other = B.HEX_EDGES[other_hex].index(e)
    raw = {
        "hexes": HEXES,
        "players": [{"color": "red",
                     "settlements": [hex_ref(9, 0, "corner"), hex_ref(4, h4, "corner"), hex_ref(5, h5, "corner")],
                     "roads": [hex_ref(9, 0, "side"), hex_ref(other_hex, k_other, "side")]}],
    }
    parsed = llm.convert_pieces(raw)
    assert parsed["players"][0]["settlements"] == [v]
    assert parsed["players"][0]["roads"] == [e]


def test_convert_pieces_accepts_bare_ids_and_lists_and_reports_bad_entries():
    warnings: list = []
    raw = {
        "hexes": HEXES,
        "players": [{"color": "red",
                     "settlements": [17, [9, 0], {"hex": 40, "corner": 0}, {"hex": 1, "corner": 9}, "junk"],
                     "cities": [{"hex": 9, "corner": 0}],   # same vertex as a settlement -> city wins
                     "roads": [3, {"hex": 0}]}],
        "ports": [{"edge": 5, "type": "ore"}, {"hex": 99, "side": 1, "type": "3:1"}],
    }
    parsed = llm.convert_pieces(raw, warnings)
    p = parsed["players"][0]
    assert p["settlements"] == [17]
    assert p["cities"] == [B.HEX_VERTICES[9][0]]
    assert p["roads"] == [3]
    assert parsed["ports"] == [{"edge": 5, "type": "ore"}]
    assert len(warnings) == 5  # hex 40, corner 9, "junk", {"hex": 0}, port hex 99
    assert any("hex 40" in w for w in warnings)
    assert any("corner 9" in w for w in warnings)
    # input not mutated
    assert raw["players"][0]["settlements"][1] == [9, 0]


# ---------------------------------------------------------------------------
# prompt / tool definition
# ---------------------------------------------------------------------------
def test_build_prompt_explains_indexing():
    prompt = llm.build_prompt()
    assert "clockwise" in prompt.lower()
    for phrase in ("0..18", "corner 0 = top", "side k", "corner k and corner k+1",
                   llm.TOOL_NAME, "3:1", "confidence", "12..15", "16,17,18"):
        assert phrase in prompt, phrase
    for key in llm.CONFIDENCE_KEYS:
        assert key in prompt


def test_build_tool_schema_is_derived_from_parse_schema():
    tool = llm.build_tool()
    assert tool["name"] == llm.TOOL_NAME
    schema = tool["input_schema"]
    assert "$schema" not in schema and "title" not in schema
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"hexes", "players", "confidence"}
    # hexes untouched from PARSE_SCHEMA
    assert schema["properties"]["hexes"] == S.PARSE_SCHEMA["properties"]["hexes"]
    pp = schema["properties"]["players"]["items"]["properties"]
    assert pp["settlements"]["items"]["required"] == ["hex", "corner"]
    assert pp["cities"]["items"]["required"] == ["hex", "corner"]
    assert pp["roads"]["items"]["required"] == ["hex", "side"]
    assert pp["settlements"]["items"]["properties"]["hex"]["maximum"] == B.NUM_HEXES - 1
    assert pp["roads"]["items"]["properties"]["side"]["maximum"] == 5
    ports = schema["properties"]["ports"]["items"]
    assert set(ports["required"]) == {"hex", "side", "type"}
    conf = schema["properties"]["confidence"]
    assert set(conf["properties"]) == set(llm.CONFIDENCE_KEYS)
    # PARSE_SCHEMA itself must not have been modified
    assert S.PARSE_SCHEMA["properties"]["players"]["items"]["properties"]["settlements"]["items"]["type"] == "integer"
    assert "confidence" not in S.PARSE_SCHEMA["properties"]


# ---------------------------------------------------------------------------
# image handling
# ---------------------------------------------------------------------------
def test_encode_image_downsizes_to_max_side_and_base64_png():
    media, data, size = llm.encode_image(sample_image((3136, 1600)))
    assert media == "image/png"
    assert size == (1568, 800)
    decoded = Image.open(io.BytesIO(base64.standard_b64decode(data)))
    assert decoded.size == (1568, 800)
    # small images are not upscaled
    media, data, size = llm.encode_image(sample_image((640, 400)))
    assert size == (640, 400)


def test_load_image_from_path_bytes_and_pil(tmp_path):
    img = sample_image()
    path = tmp_path / "shot.png"
    img.save(path)
    assert llm.load_image(str(path)).size == img.size
    assert llm.load_image(path).size == img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert llm.load_image(buf.getvalue()).size == img.size
    assert llm.load_image(img.convert("RGBA")).mode == "RGB"
    with pytest.raises(FileNotFoundError):
        llm.load_image(tmp_path / "missing.png")


# ---------------------------------------------------------------------------
# end-to-end with a fake client
# ---------------------------------------------------------------------------
def test_parse_with_claude_fake_client_builds_state(tmp_path):
    path = tmp_path / "shot.png"
    sample_image((2000, 1200)).save(path)
    client = FakeClient(tool_use_response(make_tool_input()))

    result = llm.parse_with_claude(path, client=client)

    assert isinstance(result, ParseResult)
    # --- request shape --------------------------------------------------
    assert len(client.messages.calls) == 1
    req = client.messages.calls[0]
    assert req["model"] == llm.DEFAULT_MODEL == "claude-opus-5"
    assert req["tool_choice"] == {"type": "tool", "name": llm.TOOL_NAME}
    assert [t["name"] for t in req["tools"]] == [llm.TOOL_NAME]
    assert "corner 0 = top" in req["system"]
    msg = req["messages"][0]
    assert msg["role"] == "user"
    img_block, text_block = msg["content"]
    assert img_block["type"] == "image" and img_block["source"]["type"] == "base64"
    assert img_block["source"]["media_type"] == "image/png"
    sent = Image.open(io.BytesIO(base64.standard_b64decode(img_block["source"]["data"])))
    assert max(sent.size) == 1568
    assert text_block["type"] == "text"
    assert "max_tokens" in req and "thinking" not in req

    # --- converted parse -------------------------------------------------
    red = result.parsed["players"][0]
    assert red["settlements"] == [B.HEX_VERTICES[9][0], B.HEX_VERTICES[0][3]]
    assert red["roads"][0] == B.HEX_EDGES[0][5]
    assert "confidence" not in result.parsed

    # --- state -------------------------------------------------------------
    st = result.state
    assert st.phase == PHASE_MAIN
    assert [p.color for p in st.players] == ["red", "blue"]
    assert st.players[0].settlements == [B.HEX_VERTICES[9][0], B.HEX_VERTICES[0][3]]
    assert st.players[0].cities == [B.HEX_VERTICES[16][4]]
    assert st.players[0].roads == [B.HEX_EDGES[0][5], B.HEX_EDGES[9][0], B.HEX_EDGES[16][4]]
    assert st.players[1].settlements == [B.HEX_VERTICES[2][1], B.HEX_VERTICES[11][2]]
    assert st.robber == 9
    assert st.hexes == list(B.STANDARD_HEXES)
    assert st.players[0].hand_known and st.players[0].resources == [2, 1, 0, 1, 1]
    assert not st.players[1].hand_known and st.players[1].hand_size == 4
    assert st.current == 1 and st.dice == 8
    assert st.ports[B.EDGE_VERTICES[B.HEX_EDGES[0][5]][0]] == B.PORT_GENERIC
    assert st.ports[B.EDGE_VERTICES[B.HEX_EDGES[2][0]][0]] == B.WOOD

    # --- confidence / debug ----------------------------------------------
    assert result.confidence["robber"] == 1.0 and result.confidence["overall"] == 0.8
    assert result.debug["usage"] == {"input_tokens": 1234, "output_tokens": 567}
    assert result.debug["stop_reason"] == "tool_use"
    assert result.debug["raw"]["robber"] == 9
    assert result.debug["image_size"] == [1568, 941]
    # only the model's warnings from validate (a pieceless position is fine, ports count is not 9)
    assert any("9 ports" in w for w in result.warnings)


def test_parse_with_claude_me_override_and_pil_input():
    client = FakeClient(tool_use_response(make_tool_input()))
    result = llm.parse_with_claude(sample_image(), me="Blue", client=client)
    assert result.parsed["me"] == "blue"
    # blue reported no exact hand, so even as "me" only the card count is known
    assert not result.state.players[1].hand_known and result.state.players[1].hand_size == 4
    assert not result.state.players[0].hand_known  # red is no longer "me"
    assert "blue" in client.messages.calls[0]["messages"][0]["content"][1]["text"]

    client = FakeClient(tool_use_response(make_tool_input()))
    result = llm.parse_with_claude(sample_image(), me="green", client=client)
    assert any("green" in w for w in result.warnings)


def test_parse_with_claude_effort_and_model_passthrough():
    client = FakeClient(tool_use_response(make_tool_input()))
    llm.parse_with_claude(sample_image(), client=client, model="claude-sonnet-5", effort="low", max_tokens=4000)
    req = client.messages.calls[0]
    assert req["model"] == "claude-sonnet-5"
    assert req["output_config"] == {"effort": "low"}
    assert req["max_tokens"] == 4000


def test_models_without_forced_tool_choice_use_auto():
    client = FakeClient(tool_use_response(make_tool_input()))
    llm.parse_with_claude(sample_image(), client=client, model="claude-fable-5-1")
    assert client.messages.calls[0]["tool_choice"]["type"] == "auto"


def test_dict_response_and_text_json_fallback():
    # plain dict blocks (e.g. from response.to_dict()) are accepted
    resp = {"content": [{"type": "tool_use", "name": llm.TOOL_NAME, "input": make_tool_input()}],
            "stop_reason": "tool_use"}
    result = llm.parse_with_claude(sample_image(), client=FakeClient(resp))
    assert result.state.players[0].cities == [B.HEX_VERTICES[16][4]]

    # model answered in text with a fenced JSON object -> used with a warning
    import json
    text = "Here is the board:\n```json\n" + json.dumps(make_tool_input()) + "\n```"
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                           stop_details=None, model="m", usage=None)
    result = llm.parse_with_claude(sample_image(), client=FakeClient(resp))
    assert result.state.robber == 9
    assert any("JSON text" in w for w in result.warnings)


def test_missing_confidence_and_missing_me_produce_warnings():
    raw = make_tool_input()
    del raw["confidence"]
    del raw["me"]
    result = llm.parse_with_claude(sample_image(), client=FakeClient(tool_use_response(raw)))
    assert result.confidence == {}
    assert any("confidence" in w for w in result.warnings)
    assert any("screen owner" in w for w in result.warnings)


def test_no_report_and_refusal_raise():
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="I cannot see a board.")],
                           stop_reason="end_turn", stop_details=None, model="m", usage=None)
    with pytest.raises(RuntimeError, match="did not return"):
        llm.parse_with_claude(sample_image(), client=FakeClient(resp))

    resp = SimpleNamespace(content=[], stop_reason="max_tokens", stop_details=None, model="m", usage=None)
    with pytest.raises(RuntimeError, match="max_tokens"):
        llm.parse_with_claude(sample_image(), client=FakeClient(resp))

    resp = SimpleNamespace(content=[], stop_reason="refusal",
                           stop_details=SimpleNamespace(category="other", explanation="policy"),
                           model="m", usage=None)
    with pytest.raises(RuntimeError, match="declined.*policy"):
        llm.parse_with_claude(sample_image(), client=FakeClient(resp))


# ---------------------------------------------------------------------------
# package / key / API error handling
# ---------------------------------------------------------------------------
def test_missing_anthropic_package_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` raise ImportError
    with pytest.raises(RuntimeError, match="pip install anthropic"):
        llm.parse_with_claude(sample_image())


def _fake_anthropic_module(client_factory=None):
    mod = types.ModuleType("anthropic")

    class AnthropicError(Exception):
        pass

    class APIError(AnthropicError):
        def __init__(self, message, status_code=None):
            super().__init__(message)
            self.message = message
            self.status_code = status_code

    class APIConnectionError(APIError):
        pass

    class APIStatusError(APIError):
        pass

    class Anthropic:
        def __init__(self, api_key=None):
            if client_factory is None:
                raise AnthropicError("The api_key client option must be set")
            self.messages = client_factory(api_key).messages

    mod.AnthropicError = AnthropicError
    mod.APIError = APIError
    mod.APIConnectionError = APIConnectionError
    mod.APIStatusError = APIStatusError
    mod.Anthropic = Anthropic
    return mod


def test_missing_api_key_gives_export_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module())
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="export ANTHROPIC_API_KEY="):
        llm.parse_with_claude(sample_image())


def test_api_key_from_argument_and_env_builds_client(monkeypatch):
    seen: list = []

    def factory(api_key):
        seen.append(api_key)
        return FakeClient(tool_use_response(make_tool_input()))

    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(factory))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = llm.parse_with_claude(sample_image(), api_key="sk-ant-test")
    assert result.state.robber == 9
    assert seen == ["sk-ant-test"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    llm.parse_with_claude(sample_image())
    assert seen[-1] is None  # the SDK reads the env var itself


def test_api_errors_are_wrapped_in_runtime_error(monkeypatch):
    mod = _fake_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    err = mod.APIStatusError("overloaded", status_code=529)
    with pytest.raises(RuntimeError, match="server error 529"):
        llm.parse_with_claude(sample_image(), client=FakeClient(error=err))
    err = mod.APIStatusError("invalid x-api-key", status_code=401)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.parse_with_claude(sample_image(), client=FakeClient(error=err))
    err = mod.APIConnectionError("Connection error.")
    with pytest.raises(RuntimeError, match="could not reach"):
        llm.parse_with_claude(sample_image(), client=FakeClient(error=err))


def test_forced_tool_choice_rejection_retries_with_auto(monkeypatch):
    mod = _fake_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", mod)

    class FlakyMessages(FakeMessages):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["tool_choice"]["type"] == "tool":
                raise mod.APIStatusError('tool_choice: type "tool" is not supported for this model.',
                                         status_code=400)
            return tool_use_response(make_tool_input())

    client = FakeClient()
    client.messages = FlakyMessages()
    result = llm.parse_with_claude(sample_image(), client=client)
    assert result.state.robber == 9
    assert [c["tool_choice"]["type"] for c in client.messages.calls] == ["tool", "auto"]


# ---------------------------------------------------------------------------
# Regression tests for the robustness review (vision-io-1, vision-io-4, vision-io-6)
# ---------------------------------------------------------------------------
def test_unknown_me_never_reaches_the_state(monkeypatch):
    """vision-io-1: a --me colour that is not a detected player must not propagate (KeyError in the CLI)."""
    result = llm.parse_with_claude(sample_image(), me="green", client=FakeClient(tool_use_response(make_tool_input())))
    assert result.parsed["me"] == "red"          # the model's answer is kept
    assert any("green" in w and "using 'red'" in w for w in result.warnings)
    result.state.player_index(result.parsed["me"])   # what cli.cmd_analyze does
    # the model itself reports an unknown owner -> first player, with a warning
    raw = make_tool_input()
    raw["me"] = "purple"
    result = llm.parse_with_claude(sample_image(), client=FakeClient(tool_use_response(raw)))
    assert result.parsed["me"] == "red"
    assert any("purple" in w for w in result.warnings)
    # no players at all: no crash, no me
    raw = make_tool_input()
    raw["players"] = []
    result = llm.parse_with_claude(sample_image(), me="red", client=FakeClient(tool_use_response(raw)))
    assert result.state.num_players == 0


def test_me_override_drops_mis_attributed_hand():
    client = FakeClient(tool_use_response(make_tool_input()))
    result = llm.parse_with_claude(sample_image(), me="blue", client=client)
    assert result.parsed["me"] == "blue"
    assert "resources" not in result.parsed["players"][0]
    assert any("attributed the hand bar" in w for w in result.warnings)
    assert not result.state.players[0].hand_known


def test_rolled_flag_in_prompt_schema_and_state():
    """vision-io-4: the model can say whether the roll already happened."""
    assert "rolled" in llm.build_prompt()
    assert "rolled" in llm.build_tool_schema()["properties"]
    raw = make_tool_input()
    raw["rolled"] = False          # dice 8 still displayed from the previous turn
    result = llm.parse_with_claude(sample_image(), client=FakeClient(tool_use_response(raw)))
    assert result.state.phase == "roll" and result.state.dice == 8
    raw = make_tool_input()
    del raw["dice"]
    result = llm.parse_with_claude(sample_image(), client=FakeClient(tool_use_response(raw)))
    assert result.state.phase == "roll"


def test_sdk_resolved_credentials_are_used_without_env_vars(monkeypatch):
    """vision-io-6: no environment pre-check; the SDK may hold a stored profile / federation credentials."""
    seen: list = []

    def factory(api_key):
        seen.append(api_key)
        return FakeClient(tool_use_response(make_tool_input()))

    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(factory))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    result = llm.parse_with_claude(sample_image())
    assert result.state.robber == 9 and seen == [None]
    # and when the SDK cannot resolve anything, the hint mentions both ways to log in
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module())
    with pytest.raises(RuntimeError, match="ant auth login"):
        llm.parse_with_claude(sample_image())
