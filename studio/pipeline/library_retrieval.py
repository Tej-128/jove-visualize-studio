"""Restrict every retrieval channel BEFORE ranking to actually available media.

The original Visualizer engine is called unchanged. Filtering its final top hits
was insufficient: unavailable videos could crowd all usable candidates out.
"""
import sqlite3
import numpy as np


class LibraryCorpusIndex:
    def __init__(self,index,available_ids):
        self.base=index;self.allowed=sorted(set(map(str,available_ids)))
        self.marks=','.join('?' for _ in self.allowed);self.titles=None
        rows=index.connection.execute(f'SELECT passage_id FROM passages WHERE article_id IN ({self.marks}) ORDER BY passage_id',self.allowed).fetchall() if self.allowed else []
        self.passage_ids=np.array([r[0] for r in rows],dtype=np.int64)
        self.vectors=np.asarray(index.embeddings[self.passage_ids-1],dtype=np.float32)

    def __getattr__(self,name):return getattr(self.base,name)

    def retrieve(self,step):
        from jove_visualizer.retrieval import CorpusIndex
        return CorpusIndex.retrieve(self,step)

    def _semantic_top_k(self,texts,top_k):
        query=np.asarray(self.embedder.encode_queries(texts,batch_size=max(1,len(texts))),dtype=np.float32)
        scores=self.vectors@query.T;result=[]
        for column in scores.T:
            k=min(top_k,len(column))
            positions=np.argsort(column)[::-1][:k] if k else []
            result.append([(int(self.passage_ids[i]),float(column[i])) for i in positions])
        return result,query

    def _lexical_top_k(self,text,top_k):
        from jove_visualizer.features import build_fts_query
        query=build_fts_query(text)
        if not query or not self.allowed:return []
        rows=self.connection.execute(f'''SELECT passage_fts.rowid AS passage_id,bm25(passage_fts) AS score
            FROM passage_fts JOIN passages p ON p.passage_id=passage_fts.rowid
            WHERE passage_fts MATCH ? AND p.article_id IN ({self.marks}) ORDER BY score LIMIT ?''',
            [query,*self.allowed,top_k]).fetchall()
        return [(int(r[0]),1/(1+rank)) for rank,r in enumerate(rows)]

    def _load_article_title_cache(self):
        if self.titles is None:
            ids,titles,vectors,_=self.base._load_article_title_cache()
            positions=[i for i,vid in enumerate(ids) if vid in self.allowed]
            kept=[ids[i] for i in positions]
            self.titles=(kept,[titles[i] for i in positions],np.asarray(vectors[positions]),{v:i for i,v in enumerate(kept)})
        return self.titles


def restrict_index(index,available_ids):
    # Test doubles and alternative index adapters need not be sqlite-backed.
    return LibraryCorpusIndex(index,available_ids) if isinstance(index.connection,sqlite3.Connection) else index
