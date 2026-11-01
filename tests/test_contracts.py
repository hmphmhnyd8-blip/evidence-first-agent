import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from build_database import build_database
from sqlite_repository import query_ship_speed_sqlite, list_ship_ids_sqlite
from tool_runtime import execute_tool
import evidence_graph as eg
import agent_graph as ag
import evidence_tool as et


@pytest.fixture
def db(tmp_path):
    path = tmp_path / 'demo.db'
    build_database([{'vessel_id':'A','sog':0}, {'vessel_id':'A','sog':10}, {'vessel_id':'A','sog':None}, {'vessel_id':'B','sog':None}], path)
    return path


def test_zero_is_valid_and_missing_is_not_zero(db):
    result = query_ship_speed_sqlite(db, 'A')
    assert result['statistics'] == dict(total_records=3,valid_records=2,average_speed=5,max_speed=10)


def test_absent_and_all_missing_are_different(db):
    assert query_ship_speed_sqlite(db,'B')['status']=='no_valid_speed'
    assert query_ship_speed_sqlite(db,'C') == {'vessel_id':'C','status':'not_found','statistics':None}


def test_parameterized_query_and_no_database_mutation(db):
    before=db.read_bytes()
    assert query_ship_speed_sqlite(db,"A' OR 1=1 --")['status']=='not_found'
    assert list_ship_ids_sqlite(db)==['A','B']
    assert db.read_bytes()==before


def test_missing_database_does_not_create_file(tmp_path):
    missing=tmp_path/'missing.db'
    with pytest.raises(FileNotFoundError): query_ship_speed_sqlite(missing,'A')
    assert not missing.exists()


@pytest.mark.parametrize('value', ['', '   ', 123])
def test_bad_identifier(db,value):
    with pytest.raises((TypeError,ValueError)): query_ship_speed_sqlite(db,value)


def test_unknown_tool_is_not_executed():
    assert execute_tool({'name':'unregistered','args':{}}, {})['error_type']=='unknown_tool'


def test_execution_failure_is_not_empty_data():
    broken=Mock()
    broken.invoke.side_effect=FileNotFoundError('missing')
    assert execute_tool({'name':'query','args':{}},{'query':broken})['status']=='tool_error'


def state():
    return dict(question='规则？', query='规则？', matches=[],decision={},rewrite_count=0,llm_calls=0)


def decision(ok=False, ids=None):
    return dict(sufficient=ok,evidence_ids=ids or [],reason='test',missing_information='' if ok else '缺少依据')


def model_returning(value):
    model=Mock()
    model.with_structured_output.return_value.invoke.return_value=eg.EvidenceCheck(**value)
    return model


def test_empty_evidence_does_not_call_model():
    model=Mock()
    assert not eg.check_evidence(model,'?',[])['sufficient']
    model.with_structured_output.assert_not_called()


@pytest.mark.parametrize('value',[decision(True,['invented']),decision(True)])
def test_invalid_evidence_ids_are_rejected(value):
    with pytest.raises(ValueError): eg.check_evidence(model_returning(value),'?',[{'chunk_id':'real','text':'rule'}])


def test_insufficient_evidence_rewrites_only_once(monkeypatch):
    retrieval=Mock()
    retrieval.invoke.return_value={'matches':[{'chunk_id':'one','text':'rule'}]}
    monkeypatch.setattr(eg,'search_documents',retrieval)
    monkeypatch.setattr(eg,'check_evidence',lambda **kw: decision())
    rewrite=Mock(return_value='补充查询')
    monkeypatch.setattr(eg,'rewrite_query',rewrite)
    result=eg.build_evidence_graph(Mock()).invoke(state())
    assert result['rewrite_count']==1 and result['llm_calls']==3
    assert len(result['matches'])==1 and retrieval.invoke.call_count==2
    rewrite.assert_called_once()


def test_budget_stops_before_rewrite(monkeypatch):
    retrieval=Mock()
    retrieval.invoke.return_value={'matches':[{'chunk_id':'one','text':'rule'}]}
    monkeypatch.setattr(eg,'search_documents',retrieval)
    monkeypatch.setattr(eg,'check_evidence',lambda **kw: decision())
    rewrite=Mock()
    monkeypatch.setattr(eg,'rewrite_query',rewrite)
    with pytest.raises(eg.ModelCallLimitError): eg.build_evidence_graph(Mock(),max_llm_calls=1).invoke(state())
    rewrite.assert_not_called()


def test_evidence_exception_is_tool_error(monkeypatch):
    graph=Mock()
    graph.invoke.side_effect=RuntimeError('failure')
    monkeypatch.setattr(et,'build_evidence_graph',lambda *a,**k:graph)
    assert et.make_evidence_search(Mock()).invoke({'question':'规则'})['status']=='tool_error'


def test_outer_loop_is_bounded_and_pairs_tool_messages(monkeypatch):
    model=Mock()
    model.invoke.side_effect=[AIMessage(content='',tool_calls=[dict(name='query',args={},id=str(i))]) for i in range(6)]
    tool=Mock()
    tool.invoke.return_value={'status':'ok'}
    monkeypatch.setattr(ag,'save_tool_log',lambda *args:None)
    messages=[{'role':'user','content':'query'}]
    ag.run_agent_turn(model,messages,{'query':tool})
    assert model.invoke.call_count==6 and tool.invoke.call_count==6
    assert '上限' in messages[-1].content
    assert [m.tool_call_id for m in messages if m.type=='tool']==[str(i) for i in range(6)]
