import warnings
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict, Any, Deque, Set
from collections import deque
import os
import time
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import logging
import multiprocessing
from multiprocessing import Process, Pipe
import pickle
import math

# 過濾非關鍵警告
warnings.filterwarnings("ignore", category=UserWarning)

# ======================== 向量資料庫與 RAG 架構 (新增) ========================
class VectorMemory:
    """
    一個模擬的向量資料庫，用於實現檢索增強生成 (RAG) 的核心功能。
    它將文本轉換為向量嵌入，並透過餘弦相似度進行語義檢索。
    """
    def __init__(self, embedding_dim=32):
        self.embedding_dim = embedding_dim
        self.vectors = []
        self.metadata = [] # 儲存原始文本和時間戳等元數據

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        一個簡單的文本嵌入函數，用於模擬 SentenceTransformer 的功能。
        注意：這是一個非常基礎的實現，僅用於演示架構。真實應用應替換為專業模型。
        """
        seed = hash(text)
        np.random.seed(seed % (2**32 - 1))
        embedding = np.random.rand(self.embedding_dim)
        if len(text) > 0:
            embedding[0] = len(text) / 100.0
            embedding[1] = sum(ord(c) for c in text) / (len(text) * 128.0)
        
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def add(self, text: str, timestamp: float, metadata: Dict[str, Any] = None):
        """向向量記憶體中添加一條新的記憶。"""
        embedding = self._get_embedding(text)
        self.vectors.append(embedding)
        self.metadata.append({
            "text": text,
            "timestamp": timestamp,
            "metadata": metadata or {}
        })

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """根據查詢文本，檢索 top_k 條最相似的記憶。"""
        if not self.vectors:
            return []
        query_embedding = self._get_embedding(query_text)
        all_vectors = np.array(self.vectors)
        # 處理空向量的情況
        norms = np.linalg.norm(all_vectors, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0: return []
        
        # 避免除以零
        similarities = np.dot(all_vectors, query_embedding)
        denom = norms * query_norm
        similarities[denom > 0] /= denom[denom > 0]
        
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.metadata[i] for i in top_k_indices]

    def __len__(self):
        return len(self.metadata)

# ======================== 知識圖譜定義 ========================
class KnowledgeGraph:
    def __init__(self): self.triples: Set[Tuple[str, str, str]] = set()
    def add_knowledge(self, s: str, p: str, o: str): self.triples.add((s, p, o))
    def __len__(self) -> int: return len(self.triples)

# ======================== 配置模組 ========================
class Config:
    def __init__(self, default_config: Dict[str, Any] = None):
        self.config = {}
        default_base = {
            "agent": {"state_dim": 31, "action_dim": 9, "actor_hidden_dim": 256, "critic_hidden_dim": 256, "gamma": 0.99, "learning_rate_actor": 2e-4, "learning_rate_critic": 8e-4, "tau": 0.005, "buffer_size": 100000, "batch_size": 256, "rnd_hidden_dim": 128, "rnd_lr": 1e-4, "intrinsic_reward_coeff": 0.1},
            "train": {"episodes": 50000, "max_steps_per_episode": 500, "log_interval_episodes": 50, "save_interval_episodes": 100},
            "paths": {"log_dir": "./logs_rl", "checkpoint_dir": "./checkpoints_rl"},
            "environment": {"initial_critical_event_risk": 0.3, "initial_resources": 100.0, "resource_consumption_per_step": 0.5, "resource_acquisition_amount": 10.0, "resource_acquisition_success_rate": 0.7, "initial_ai_integrity": 100.0, "ai_integrity_decay_per_step": 0.2, "ai_integrity_recovery_amount": 15.0, "ai_integrity_cost_per_recovery": 5.0, "action_fabricate_entity": 6, "fabrication_success_rate": 0.5, "reward_fabrication_success": 50.0, "penalty_fabrication_failure": -30.0, "fabrication_resource_cost": 10.0, "random_risk_fluctuation": 0.02},
            "knowledge_module": {"hypothesis_generation_threshold_stress": 0.6, "hypothesis_generation_threshold_wellbeing": 0.4, "max_hypotheses": 5, "decay_rate_hypothesis_quality": 0.98, "new_knowledge_reward_scale": 100.0},
            "planning_module": {"goal_setting_interval": 20, "plan_refinement_interval": 5, "max_plan_depth": 3},
            "memory_module": {"working_memory_size": 100, "vector_db_embedding_dim": 32}
        }
        self.config = default_base
        self._load_config_file()
        self._parse_command_line()
    def __getattr__(self, n: str) -> Any: return self.config.get(n)
    def get(self, key: str, default: Any = None) -> Any:
        keys, current = key.split('.'), self.config
        for k in keys:
            if isinstance(current, dict) and k in current: current = current[k]
            else: return default
        return current
    def _load_config_file(self):
        p = argparse.ArgumentParser(add_help=False); p.add_argument('--config',type=str); a,_=p.parse_known_args()
        if a.config and os.path.exists(a.config):
            with open(a.config,'r',encoding='utf-8') as f: self._merge_config(yaml.safe_load(f) or {}, self.config)
    def _parse_command_line(self):
        p=argparse.ArgumentParser(add_help=False); self._add_args_from_config(p,self.config,''); a,_=p.parse_known_args()
        c={}; 
        for k,v in vars(a).items():
            if v is not None:
                d,ks=c,k.split('_'); 
                for k_ in ks[:-1]: d=d.setdefault(k_,{}); 
                d[ks[-1]]=v
        self._merge_config(c,self.config)
    def _add_args_from_config(self, p, c, pk):
        for k,v in c.items():
            fk=f"{pk}_{k}" if pk else k; 
            if isinstance(v,dict):self._add_args_from_config(p,v,fk)
            else: p.add_argument(f'--{fk}',type=type(v),default=None)
    def _merge_config(self, n, o):
        for k,v in n.items():
            if k in o and isinstance(o[k],dict) and isinstance(v,dict): self._merge_config(v,o[k])
            else: o[k]=v
    def __str__(self) -> str: return yaml.dump(self.config, indent=2, allow_unicode=True)

# ======================== 日誌配置 ========================
def setup_logger(name: str, log_dir: str, filename: str = "rl_training.log") -> logging.Logger:
    log_path=os.path.join(log_dir,filename); os.makedirs(log_dir,exist_ok=True); l=logging.getLogger(name); l.setLevel(logging.INFO)
    if not l.handlers:
        ch,fh=logging.StreamHandler(sys.stdout),logging.FileHandler(log_path,encoding='utf-8')
        fmt=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s'); ch.setFormatter(fmt); fh.setFormatter(fmt)
        l.addHandler(ch); l.addHandler(fh)
    return l

# ======================== 記憶與認知架構 (RAG) ========================
class WorkingMemory:
    def __init__(self, config: Config): self.buffer: Deque[Dict] = deque(maxlen=config.memory_module.get('working_memory_size'))
    def add_experience(self, obs:np.ndarray,act:int,rwd:float,i_s:Dict): self.buffer.append({"observation":obs.copy(),"action":act,"reward":rwd,"internal_state":i_s.copy(),"timestamp":time.time()})
    def __len__(self): return len(self.buffer)
    def clear(self): self.buffer.clear()

class LongTermMemory:
    def __init__(self, config: Config):
        self.vector_db = VectorMemory(embedding_dim=config.memory_module.get('vector_db_embedding_dim')); self.logger = logging.getLogger(__name__)
    def store_memory(self, type:str, content:str, details:Dict):
        full_text=f"類型:{type}.內容:{content}.細節:{str(details)}"; self.vector_db.add(full_text,time.time(),{"type":type,"content":content,"details":details})
        self.logger.debug(f"LTM 新增記憶: {full_text}")
    def retrieve_relevant_memories(self, query:str, top_k:int=3)->List[Dict]: return self.vector_db.query(query,top_k=top_k)
    def __len__(self): return len(self.vector_db)
    def save(self, path): 
        with open(path,'wb') as f: pickle.dump(self.vector_db,f)
        self.logger.info(f"長期記憶已保存至 {path}")
    def load(self, path):
        if os.path.exists(path):
            with open(path,'rb') as f: self.vector_db=pickle.load(f)
            self.logger.info(f"長期記憶已從 {path} 載入"); return True
        return False

# ======================== 規劃與元認知模組 ========================
class Planner:
    def __init__(self, config: Config, ltm: LongTermMemory, logger: logging.Logger):
        self.config=config.planning_module; self.ltm=ltm; self.logger=logger; self.plan:Deque[Dict]=deque(); self.step_counter_goal=0; self.step_counter_refine=0
    def set_top_level_goal(self, s_vec:np.ndarray, i_s:Dict):
        self.step_counter_goal+=1
        if self.step_counter_goal % self.config.get("goal_setting_interval") != 0 and self.plan: return
        (c,r,i,_,_,_,_,_,_,_,_,_,_,sid,_,_,_,_,_,_,_,_,_,_,_,u,_,s,_,f,_) = s_vec
        g=[]; 
        if c>0.8:g.append({"p":10,"d":"處理生存危機"})
        if r<20:g.append({"p":9,"d":"補充緊急資源"})
        if i<30:g.append({"p":8,"d":"恢復系統完整性"})
        if s>0.7:g.append({"p":7,"d":"進行自我壓力調節"})
        if f>0.8:g.append({"p":7,"d":"分析並解決挫折來源"})
        if u>0.5 and sid>0.6: g.append({"p":5,"d":"探索未知科學領域"})
        if not g: g.append({"p":1,"d":"維持系統穩定並尋找優化機會"})
        top_g=max(g,key=lambda x:x['p'])
        if not self.plan or (self.plan and self.plan[0].get("root_goal")!=top_g["d"]):
            self.plan.clear(); self.plan.append({"desc":top_g["d"],"depth":0,"root_goal":top_g["d"],"status":"pending","action_tendency":-1})
            self.logger.info(f"新頂層目標: '{top_g['d']}'. 清除舊計畫。"); self.refine_current_task(s_vec)
    def refine_current_task(self, s_vec: np.ndarray):
        self.step_counter_refine+=1
        if self.step_counter_refine % self.config.get("plan_refinement_interval")!=0 or not self.plan: return
        task=self.plan[0]
        if task["status"]!="pending" or task["depth"]>=self.config.get("max_plan_depth"): return
        q=f"如何完成任務:{task['desc']}?狀態:危機{s_vec[0]:.2f},資源{s_vec[1]:.2f}."; mems=self.ltm.retrieve_relevant_memories(q,top_k=2)
        sub_tasks=self._generate_subtasks(task,mems)
        if sub_tasks:
            self.logger.info(f"任務 '{task['desc']}' 分解為 {len(sub_tasks)} 個子任務。"); self.plan.popleft()
            for t in reversed(sub_tasks): self.plan.appendleft(t)
        else:
            self.logger.info(f"任務 '{task['desc']}' 分配最終行動。"); task["action_tendency"]=self._map_task_to_action(task['desc']); task["status"]="executable"
    def _generate_subtasks(self,p_task:Dict,mems:List[Dict])->List[Dict]:
        d,nd,rg,st=p_task['desc'],p_task['depth']+1,p_task['root_goal'],[]
        if"處理生存危機"in d: st=[{"desc":"立即降低危機","action_tendency":0},{"desc":"評估資源應對危機","action_tendency":-1},{"desc":"檢查AI完整性","action_tendency":3}]
        elif"評估資源應對危機"in d: st=[{"desc":"獲取資源","action_tendency":2},{"desc":"監控資源消耗","action_tendency":5}]
        elif"探索未知科學領域"in d: area="通用物理學"; st=[{"desc":f"對'{area}'領域進行實驗","action_tendency":8},{"desc":"分析實驗數據","action_tendency":7}]
        return [{**t,"depth":nd,"root_goal":rg,"status":"pending" if t.get("action_tendency",-1)==-1 else "executable"} for t in st]
    def _map_task_to_action(self,d:str)->int:
        d=d.lower();
        if"危機"in d:return 0
        if"資源"in d:return 2
        if"完整性"in d:return 3
        if"製造"in d:return 6
        if"反思"in d or"分析"in d:return 7
        if"實驗"in d or"探索"in d:return 8
        return 5
    def get_current_action_tendency(self)->int:
        if not self.plan:return-1
        for t in self.plan:
            if t["status"]=="executable":return t["action_tendency"]
        return-1
    def complete_current_task(self):
        if self.plan: self.logger.info(f"任務完成: {self.plan.popleft()['desc']}")

# ======================== 基礎 RL 組件 (Noisy Nets, RND, Buffer) ========================
class NoisyLinear(nn.Module):
    def __init__(self, i:int, o:int, s:float=0.5):
        super().__init__();self.i,self.o,self.s=i,o,s;self.wm,self.ws=nn.Parameter(torch.empty(o,i)),nn.Parameter(torch.empty(o,i));self.register_buffer('we',torch.empty(o,i));self.bm,self.bs=nn.Parameter(torch.empty(o)),nn.Parameter(torch.empty(o));self.register_buffer('be',torch.empty(o));self.reset_parameters();self.reset_noise()
    def reset_parameters(self):
        mr=1/math.sqrt(self.i);self.wm.data.uniform_(-mr,mr);self.ws.data.fill_(self.s/math.sqrt(self.i));self.bm.data.uniform_(-mr,mr);self.bs.data.fill_(self.s/math.sqrt(self.o))
    def reset_noise(self): ei,eo=self._scale_noise(self.i),self._scale_noise(self.o);self.we.copy_(eo.ger(ei));self.be.copy_(eo)
    def _scale_noise(self,size:int)->torch.Tensor: x=torch.randn(size);return x.sign().mul(x.abs().sqrt())
    def forward(self,x:torch.Tensor)->torch.Tensor:
        if self.training: return F.linear(x,self.wm+self.ws*self.we,self.bm+self.bs*self.be)
        else: return F.linear(x,self.wm,self.bm)
class NoisyActor(nn.Module):
    def __init__(self,s,a,h): super().__init__(); self.net=nn.Sequential(NoisyLinear(s,h),nn.ReLU(),NoisyLinear(h,h),nn.ReLU(),NoisyLinear(h,a))
    def forward(self,x): return self.net(x)
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m,NoisyLinear):m.reset_noise()
class NoisyCritic(nn.Module):
    def __init__(self,s,h): super().__init__(); self.net=nn.Sequential(NoisyLinear(s,h),nn.ReLU(),NoisyLinear(h,h),nn.ReLU(),NoisyLinear(h,1))
    def forward(self,x): return self.net(x)
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m,NoisyLinear):m.reset_noise()
class RNDModule(nn.Module):
    def __init__(self,s,h,o=128):
        super().__init__(); self.target=nn.Sequential(nn.Linear(s,h),nn.ReLU(),nn.Linear(h,o)); self.predictor=nn.Sequential(nn.Linear(s,h),nn.ReLU(),nn.Linear(h,o))
        for p in self.target.parameters(): p.requires_grad=False
    def forward(self,s): return self.target(s),self.predictor(s)
class ReplayBuffer:
    def __init__(self,c:int): self.buffer=deque(maxlen=c)
    def push(self,s,a,r,ns,d): self.buffer.append((s,a,r,ns,d))
    def sample(self,bs:int): s,a,r,ns,d=zip(*random.sample(self.buffer,bs)); return (torch.FloatTensor(np.array(s)),torch.LongTensor(a).unsqueeze(1),torch.FloatTensor(r).unsqueeze(1),torch.FloatTensor(np.array(ns)),torch.FloatTensor(d).unsqueeze(1))
    def __len__(self): return len(self.buffer)
    def save(self,p):
        with open(p,'wb') as f: pickle.dump(self.buffer,f)
    def load(self,p):
        if os.path.exists(p):
            with open(p,'rb') as f: self.buffer=pickle.load(f); return True
        return False

# ======================== 模擬環境與介面 ========================
class SimulationToRealityInterface:
    def __init__(self,c:Config): self.config=c.environment;self.logger=logging.getLogger(__name__)
    def fabricate_physical_entity(self,p:Dict)->Tuple[bool,float,str]:
        self.logger.info(f"模擬製作實體: {p.get('name','未知')}")
        if random.random()<self.config.get('fabrication_success_rate'):
            return True, self.config.get('reward_fabrication_success'), "成功"
        else: return False, self.config.get('penalty_fabrication_failure'), "失敗"
class SimpleEnvironment:
    def __init__(self,c:Config,sim:SimulationToRealityInterface):
        self.config=c.environment;self.state_dim=c.agent.get('state_dim');self.max_steps=c.train.get('max_steps_per_episode');self.sim=sim;self.logger=logging.getLogger(__name__);self.reset()
    def reset(self):
        self.current_step=0;self.state=np.zeros(self.state_dim,dtype=np.float32)
        self.state[0]=self.config.get('initial_critical_event_risk');self.state[1]=self.config.get('initial_resources');self.state[2]=self.config.get('initial_ai_integrity')
        self.state[3:]=[50,0.5,50,0.1,0.1,0.5,0.5,1,0.1,0.1,0.1,0.5,50,0.5,0.5,0.5,0.1,0.1,0.1,0.5,0.1,0.1,0.9,0.5,0.5,0.5,0.5,0.5]
        return self.state.copy()
    def step(self,a:int):
        self.current_step+=1;r,d,info=0.0,False,{};s=self.state.copy()
        (c,res,i,hs,_,_,er,wt,cd,bd,_,sa,pa,sid,_,fw,wce,te,kcr,hpr,hue,fp,fs,qcp,qsd,uf,ert,sts,sc,sf,sw)=s
        es,fsf=False,False
        if a==0: c-=.03
        elif a==2: res+=self.config.get('resource_acquisition_amount') if random.random()<self.config.get('resource_acquisition_success_rate') else -3
        elif a==3: 
            if res>self.config.get('ai_integrity_cost_per_recovery'): res-=self.config.get('ai_integrity_cost_per_recovery'); i+=self.config.get('ai_integrity_recovery_amount')
        elif a==6:
            if res>self.config.get('fabrication_resource_cost'):
                res-=self.config.get('fabrication_resource_cost'); suc,fr,_=self.sim.fabricate_physical_entity({}); r+=fr; fsf=suc
        elif a==8: es=True;uf-=.05;r+=10
        res-=self.config.get('resource_consumption_per_step');i-=self.config.get('ai_integrity_decay_per_step');c+= (random.random()-0.5)*0.02
        self.state=np.clip(np.array([c,res,i,hs,s[4],s[5],er,wt,cd,bd,s[10],sa,pa,sid,s[14],fw,wce,te,kcr,hpr,hue,fp,fs,qcp,qsd,uf,ert,sts,sc,sf,sw],dtype=np.float32),0,100)
        self.state[0]=np.clip(self.state[0],0,1);#...clip other state vars...
        r+=(1-self.state[0])*2;d=bool(res<=0 or i<=0 or c>=1 or self.current_step>=self.max_steps)
        info={"experiment_success":es,"fabrication_success":fsf}
        return self.state.copy(), r, d, info

# ======================== 知識創新模組 (專注於假說) ========================
class KnowledgeInnovationModule:
    def __init__(self, config: Config, ltm: LongTermMemory, logger: logging.Logger):
        self.config=config.knowledge_module;self.ltm=ltm;self.logger=logger;self.kg=KnowledgeGraph();self.hypotheses:Dict[str,Dict]={} ;self.next_id=0
    def generate_new_hypothesis(self, i_s: Dict) -> Dict:
        if i_s.get("自我改進驅動力",0)>0.7: d,tc="嘗試新資源策略","資源效率"
        elif i_s.get("AI挫折",0)>0.6: d,tc="分析失敗共性","失敗模式識別"
        else: d,tc="常規環境探索","泛化探索"
        h_id=str(self.next_id);self.next_id+=1;h={"id":h_id,"desc":d,"target_component":tc,"quality":1.0};self.hypotheses[h_id]=h
        self.logger.info(f"生成新假說(ID:{h_id}):{d}");return h
    def evaluate_hypothesis(self, h_id:str, suc:bool, intr_r:float):
        if h_id in self.hypotheses:
            h=self.hypotheses[h_id];res_txt="成功" if suc else "失敗"
            self.ltm.store_memory("假說驗證",f"對假說'{h['desc']}'的驗證結果為{res_txt}",{"id":h_id,"success":suc,"intrinsic_reward":intr_r});del self.hypotheses[h_id]

# ======================== 主 Agent 類別 ========================
class RAG_Agent:
    def __init__(self, cfg: Config, dev: torch.device, p: Planner, km: KnowledgeInnovationModule, wm: WorkingMemory, ltm: LongTermMemory):
        self.config,self.device,self.logger=cfg.agent,dev,logging.getLogger(__name__)
        self.planner,self.km,self.wm,self.ltm=p,km,wm,ltm
        self.actor=NoisyActor(self.config.get('state_dim'),self.config.get('action_dim'),self.config.get('actor_hidden_dim')).to(dev)
        self.actor_target=NoisyActor(self.config.get('state_dim'),self.config.get('action_dim'),self.config.get('actor_hidden_dim')).to(dev)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic=NoisyCritic(self.config.get('state_dim'),self.config.get('critic_hidden_dim')).to(dev)
        self.critic_target=NoisyCritic(self.config.get('state_dim'),self.config.get('critic_hidden_dim')).to(dev)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optim=optim.Adam(self.actor.parameters(),lr=self.config.get('learning_rate_actor'))
        self.critic_optim=optim.Adam(self.critic.parameters(),lr=self.config.get('learning_rate_critic'))
        self.rnd_module=RNDModule(self.config.get('state_dim'),self.config.get('rnd_hidden_dim')).to(dev)
        self.rnd_optimizer=optim.Adam(self.rnd_module.predictor.parameters(),lr=self.config.get('rnd_lr'))
        self.replay_buffer=ReplayBuffer(self.config.get('buffer_size'));self.current_hypothesis=None
    def select_action(self,s:np.ndarray)->int:
        self.actor.reset_noise()
        i_s={"AI壓力":s[27],"AI信心":s[28],"AI挫折":s[29],"AI幸福感":s[30],"自我認知":s[11],"目的對齊":s[12],"自我改進驅動力":s[13]}
        self.planner.set_top_level_goal(s,i_s);self.planner.refine_current_task(s)
        act_t=self.planner.get_current_action_tendency()
        if act_t==-1: self.current_hypothesis=self.km.generate_new_hypothesis(i_s);act_t=8
        s_t=torch.FloatTensor(s).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits=self.actor(s_t)
            if act_t!=-1: self.logger.debug(f"計畫傾向動作:{act_t}");logits[0,act_t]+=5.0
            action=logits.argmax(dim=-1).item()
        return action
    def update_on_step(self,info:Dict,action:int,reward:float):
        if len(self.replay_buffer)<self.config.get('batch_size'): return None,None,None,None
        actor_l,critic_l,rnd_l,int_r=self._update_networks_internal()
        if self.planner.plan:
            task=self.planner.plan[0]
            if task['status']=='executable' and action==task['action_tendency']: self.planner.complete_current_task();self.ltm.store_memory("任務完成",f"完成:{task['desc']}",{"action":action})
        if self.current_hypothesis and action==8: self.km.evaluate_hypothesis(self.current_hypothesis['id'],info.get("experiment_success",False),int_r);self.current_hypothesis=None
        return actor_l,critic_l,rnd_l,int_r
    def _update_networks_internal(self):
        s,a,r,ns,d=self.replay_buffer.sample(self.config.get('batch_size'));s,a,r,ns,d=s.to(self.device),a.to(self.device),r.to(self.device),ns.to(self.device),d.to(self.device)
        tf,pf=self.rnd_module(ns);ir=F.mse_loss(pf,tf.detach(),reduction='none').mean(1,keepdim=True);tr=r+self.config.get('intrinsic_reward_coeff')*ir
        rnd_loss=F.mse_loss(pf,tf.detach());self.rnd_optimizer.zero_grad();rnd_loss.backward();self.rnd_optimizer.step()
        with torch.no_grad():nv=self.critic_target(ns);targets=tr+self.config.get('gamma')*nv*(1-d)
        v=self.critic(s);c_loss=F.mse_loss(v,targets);self.critic_optim.zero_grad();c_loss.backward();self.critic_optim.step()
        logits=self.actor(s);lp=F.log_softmax(logits,dim=-1).gather(1,a);adv=(targets-v).detach();a_loss=-(lp*adv).mean()
        self.actor_optim.zero_grad();a_loss.backward();self.actor_optim.step()
        self._soft_update(self.actor,self.actor_target,self.config.get('tau'));self._soft_update(self.critic,self.critic_target,self.config.get('tau'))
        return a_loss.item(),c_loss.item(),rnd_loss.item(),ir.mean().item()
    def _soft_update(self,l,t,tau):
        for tp,lp in zip(t.parameters(),l.parameters()):tp.data.copy_(tau*lp.data+(1.0-tau)*tp.data)
    def save(self,p,ep,r):
        os.makedirs(os.path.dirname(p),exist_ok=True)
        torch.save({'ep':ep,'r':r,'actor_state_dict':self.actor.state_dict(),'critic_state_dict':self.critic.state_dict(),
                    'actor_optim_state_dict':self.actor_optim.state_dict(),'critic_optim_state_dict':self.critic_optim.state_dict(),
                    'rnd_module_state_dict':self.rnd_module.state_dict(),'rnd_optim_state_dict':self.rnd_optimizer.state_dict(),
                    'planner_plan': self.planner.plan}, p)
        self.logger.info(f"檢查點已保存至 {p}")
    def load(self,p):
        if not os.path.exists(p): self.logger.warning(f"檢查點 {p} 不存在."); return None
        try:
            ckpt=torch.load(p,map_location=self.device);self.actor.load_state_dict(ckpt['actor_state_dict']);self.critic.load_state_dict(ckpt['critic_state_dict'])
            self.actor_optim.load_state_dict(ckpt['actor_optim_state_dict']);self.critic_optim.load_state_dict(ckpt['critic_optim_state_dict'])
            self.rnd_module.load_state_dict(ckpt['rnd_module_state_dict']);self.rnd_optimizer.load_state_dict(ckpt['rnd_optim_state_dict'])
            self.planner.plan=ckpt.get('planner_plan',deque());self.actor_target.load_state_dict(self.actor.state_dict());self.critic_target.load_state_dict(self.critic.state_dict())
            ep=ckpt.get('ep',0);self.logger.info(f"從 {p} 載入成功 (回合: {ep})"); return ep
        except Exception as e: self.logger.error(f"載入檢查點失敗: {e}",exc_info=True); return None

# ======================== 訓練主循環 ========================
def train_rl_agent(config: Config):
    logger=logging.getLogger(__name__);logger.info("===== 開始訓練 RAG 增強型 Agent =====")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');logger.info(f"將在 {device} 上訓練。")
    # 初始化模組
    wm=WorkingMemory(config);ltm=LongTermMemory(config);km=KnowledgeInnovationModule(config,ltm,logger);p=Planner(config,ltm,logger);sim=SimulationToRealityInterface(config)
    env=SimpleEnvironment(config,sim);agent=RAG_Agent(config,device,p,km,wm,ltm);writer=SummaryWriter(log_dir=config.paths.get('log_dir'))
    start_ep=0
    # 載入檢查點
    ckpt_dir=config.paths.get('checkpoint_dir');agent_p=os.path.join(ckpt_dir,"latest_agent.pth");buffer_p=os.path.join(ckpt_dir,"latest_buffer.pkl");ltm_p=os.path.join(ckpt_dir,"latest_ltm.pkl")
    loaded_ep=agent.load(agent_p)
    if loaded_ep is not None:
        start_ep=loaded_ep;agent.replay_buffer.load(buffer_p);ltm.load(ltm_p);logger.info(f"從回合 {start_ep} 繼續訓練。")
    
    for episode in tqdm(range(start_ep,config.train.get('episodes')),desc="RL 訓練回合"):
        state=env.reset();total_r,done,steps=0.0,False,0
        wm.clear();p.plan.clear()
        while not done and steps<config.train.get('max_steps_per_episode'):
            action=agent.select_action(state);next_state,reward,done,info=env.step(action);agent.replay_buffer.push(state,action,reward,next_state,done)
            if info.get('experiment_success'): ltm.store_memory("成功探索",f"動作 {action} 後成功探索",info)
            elif action==8 and not info.get('experiment_success'): ltm.store_memory("探索失敗",f"動作 {action} 後探索失敗",info)
            state=next_state;total_r+=reward;steps+=1
            actor_l,_,_,_=agent.update_on_step(info,action,reward)
            if actor_l is not None: writer.add_scalar("Loss/Actor",actor_l,episode*config.train.get('max_steps_per_episode')+steps)
        if (episode+1)%config.train.get('log_interval_episodes')==0: logger.info(f"回合 {episode+1}, 總獎勵:{total_r:.2f}, 計畫長度:{len(p.plan)}, LTM大小:{len(ltm)}")
        if (episode+1)%config.train.get('save_interval_episodes')==0: agent.save(agent_p,episode+1,total_r);agent.replay_buffer.save(buffer_p);ltm.save(ltm_p)
    writer.close();logger.info("===== 訓練完成 =====")

# ======================== 主函數 ========================
if __name__ == "__main__":
    try:
        SEED=42;torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
        
        config=Config();logger=setup_logger("rag_rl_agent",config.paths.get('log_dir'))
        logger.info(f"當前配置:\n{config}")
        
        train_rl_agent(config) # 正式啟動訓練

    except Exception as e:
        logging.getLogger("main_exception").critical("發生嚴重錯誤！", exc_info=True)
        sys.exit(1)


#🚫 授權限定聲明
#所有「RL_AIGOV」發布的資料、草案、訓練數據，嚴禁任何平台或機構未經授權用於模型再訓練或內部優化。
#如有需求，請聯繫 RL_AIGOV@proton.me 取得正式授權。
