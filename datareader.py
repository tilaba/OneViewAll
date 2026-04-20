# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import json,os,sys
import glob
import trimesh
from pathlib import Path
import pickle


# os.environ['BOP_DIR'] = '/your/path/to/bop_datasets'
BOP_LIST = ['lmo','tless','ycbv','hb','tudl','icbin','itodd', 'tyol', 'real275']
BOP_DIR = os.getenv('BOP_DIR')

def get_bop_reader(video_dir, zfar=np.inf):
  if 'ycbv' in video_dir or 'YCB' in video_dir:
    return YcbVideoReader(video_dir, zfar=zfar)
  if 'lmo' in video_dir or 'LINEMOD-O' in video_dir:
    return LinemodOcclusionReader(video_dir, zfar=zfar)
  if 'tless' in video_dir or 'TLESS' in video_dir:
    return TlessReader(video_dir, zfar=zfar)
  if 'hb' in video_dir:
    return HomebrewedReader(video_dir, zfar=zfar)
  if 'tudl' in video_dir:
    return TudlReader(video_dir, zfar=zfar)
  if 'icbin' in video_dir:
    return IcbinReader(video_dir, zfar=zfar)
  if 'itodd' in video_dir:
    return ItoddReader(video_dir, zfar=zfar)
  if 'tyol' in video_dir:
        return TyolReader(video_dir, zfar=zfar)
  if 'real275' in video_dir:
      return Real275Reader(video_dir, zfar=zfar)
  else:
      raise RuntimeError(f"Unknown dataset in path: {video_dir}")


def get_bop_video_dirs(dataset):
  if dataset=='ycbv':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/test/*'))
  elif dataset=='lmo':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/lmo/lmo_test_bop19/test/*'))
  elif dataset=='tless':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/tless/tless_test_primesense/test_primesense/*'))
  elif dataset=='hb':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/hb/hb_test_primesense_bop19/test_primesense/*'))
  elif dataset=='tudl':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/tudl/tudl_test_bop19/test/*'))
  elif dataset=='icbin':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/icbin/icbin_test_bop19/test/*'))
  elif dataset=='itodd':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/itodd/itodd_test_bop19/test/*'))
  elif dataset == 'tyol':
      # 假设路径为 $BOP_DIR/tyol/test/*
      video_dirs = sorted(glob.glob(f'{BOP_DIR}/tyol/test/*'))
  elif dataset == 'real275':
      video_dirs = sorted(glob.glob(f'{BOP_DIR}/real275/real_test/*'))
  else:
      raise RuntimeError
  return video_dirs


class YcbineoatReader:
  def __init__(self,video_dir, downscale=1, shorter_side=None, zfar=np.inf):
    self.video_dir = video_dir
    self.downscale = downscale
    self.zfar = zfar
    self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
    self.K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3,3)
    self.id_strs = []
    for color_file in self.color_files:
      id_str = os.path.basename(color_file).replace('.png','')
      self.id_strs.append(id_str)
    self.H,self.W = cv2.imread(self.color_files[0]).shape[:2]

    if shorter_side is not None:
      self.downscale = shorter_side/min(self.H, self.W)

    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.K[:2] *= self.downscale

    self.gt_pose_files = sorted(glob.glob(f'{self.video_dir}/annotated_poses/*'))

    self.videoname_to_object = {
      'bleach0': "021_bleach_cleanser",
      'bleach_hard_00_03_chaitanya': "021_bleach_cleanser",
      'cracker_box_reorient': '003_cracker_box',
      'cracker_box_yalehand0': '003_cracker_box',
      'mustard0': '006_mustard_bottle',
      'mustard_easy_00_02': '006_mustard_bottle',
      'sugar_box1': '004_sugar_box',
      'sugar_box_yalehand0': '004_sugar_box',
      'tomato_soup_can_yalehand0': '005_tomato_soup_can',
    }


  def get_video_name(self):
    return self.video_dir.split('/')[-1]

  def __len__(self):
    return len(self.color_files)

  def get_gt_pose(self,i):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      #logging.info("GT pose not found, return None")
      return None


  def get_color(self,i):
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color

  def get_mask(self,i):
    mask = cv2.imread(self.color_files[i].replace('rgb','masks'),-1)
    if len(mask.shape)==3:
      for c in range(3):
        if mask[...,c].sum()>0:
          mask = mask[...,c]
          break
    mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    return mask

  def get_depth(self,i):
    depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3
    depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    depth[(depth<0.001) | (depth>=self.zfar)] = 0
    return depth


  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.K)
    return xyz_map

  def get_occ_mask(self,i):
    hand_mask_file = self.color_files[i].replace('rgb','masks_hand')
    occ_mask = np.zeros((self.H,self.W), dtype=bool)
    if os.path.exists(hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(hand_mask_file,-1)>0)

    right_hand_mask_file = self.color_files[i].replace('rgb','masks_hand_right')
    if os.path.exists(right_hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(right_hand_mask_file,-1)>0)

    occ_mask = cv2.resize(occ_mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST)

    return occ_mask.astype(np.uint8)

  def get_gt_mesh(self):
    ob_name = self.videoname_to_object[self.get_video_name()]
    YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
    mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{ob_name}/textured_simple.obj')
    return mesh


class BopBaseReader:
  def __init__(self, base_dir, zfar=np.inf, resize=1):
    self.base_dir = base_dir
    self.resize = resize
    self.dataset_name = None
    self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*"))
    if len(self.color_files)==0:
      self.color_files = sorted(glob.glob(f"{self.base_dir}/gray/*"))
    self.zfar = zfar

    self.K_table = {}
    with open(f'{self.base_dir}/scene_camera.json','r') as ff:
      info = json.load(ff)
    for k in info:
      self.K_table[f'{int(k):06d}'] = np.array(info[k]['cam_K']).reshape(3,3)
      self.bop_depth_scale = info[k]['depth_scale']

    if os.path.exists(f'{self.base_dir}/scene_gt.json'):
      with open(f'{self.base_dir}/scene_gt.json','r') as ff:
        self.scene_gt = json.load(ff)
      self.scene_gt = copy.deepcopy(self.scene_gt)   # Release file handle to be pickle-able by joblib
      assert len(self.scene_gt)==len(self.color_files)
    else:
      self.scene_gt = None

    self.make_id_strs()


  def make_scene_ob_ids_dict(self):
    with open(f'{BOP_DIR}/{self.dataset_name}/test_targets_bop19.json','r') as ff:
      self.scene_ob_ids_dict = {}
      data = json.load(ff)
      for d in data:
        if d['scene_id']==self.get_video_id():
          id_str = f"{d['im_id']:06d}"
          if id_str not in self.scene_ob_ids_dict:
            self.scene_ob_ids_dict[id_str] = []
          self.scene_ob_ids_dict[id_str] += [d['obj_id']]*d['inst_count']
      return self.scene_ob_ids_dict


  def get_K(self, i_frame):
    K = self.K_table[self.id_strs[i_frame]]
    if self.resize!=1:
      K[:2,:2] *= self.resize
    return K


  def get_video_dir(self):
    video_id = int(self.base_dir.rstrip('/').split('/')[-1])
    return video_id

  def make_id_strs(self):
    self.id_strs = []
    for i in range(len(self.color_files)):
      name = os.path.basename(self.color_files[i]).split('.')[0]
      self.id_strs.append(name)


  def get_instance_ids_in_image(self, i_frame:int):
    ob_ids = []
    if self.scene_gt is not None:
      name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])
      for k in self.scene_gt[str(name)]:
        ob_ids.append(k['obj_id'])
    elif self.scene_ob_ids_dict is not None:
      return np.array(self.scene_ob_ids_dict[self.id_strs[i_frame]])
    else:
      mask_dir = os.path.dirname(self.color_files[0]).replace('rgb','mask_visib')
      id_str = self.id_strs[i_frame]
      mask_files = sorted(glob.glob(f'{mask_dir}/{id_str}_*.png'))
      ob_ids = []
      for mask_file in mask_files:
        ob_id = int(os.path.basename(mask_file).split('.')[0].split('_')[1])
        ob_ids.append(ob_id)
    ob_ids = np.asarray(ob_ids)
    return ob_ids


  def get_gt_mesh_file(self, ob_id):
    raise RuntimeError("You should override this")


  def get_color(self,i):
    color = imageio.imread(self.color_files[i])
    if len(color.shape)==2:
      color = np.tile(color[...,None], (1,1,3))  # Gray to RGB
    if self.resize!=1:
      color = cv2.resize(color, fx=self.resize, fy=self.resize, dsize=None)
    return color


  def get_depth(self,i, filled=False):
    if filled:
      depth_file = self.color_files[i].replace('rgb','depth_filled')
      depth_file = f'{os.path.dirname(depth_file)}/0{os.path.basename(depth_file)}'
      depth = cv2.imread(depth_file,-1)/1e3
    else:
      depth_file = self.color_files[i].replace('rgb','depth').replace('gray','depth')
      depth = cv2.imread(depth_file,-1)*1e-3*self.bop_depth_scale
    if self.resize!=1:
      depth = cv2.resize(depth, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)
    depth[depth<0.001] = 0
    depth[depth>self.zfar] = 0
    return depth

  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.get_K(i))
    return xyz_map


  def get_mask(self, i_frame:int, ob_id:int, type='mask_visib'):
    '''
    @type: mask_visib (only visible part) / mask (projected mask from whole model)
    '''
    pos = 0
    name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])
    if self.scene_gt is not None:
      for k in self.scene_gt[str(name)]:
        if k['obj_id']==ob_id:
          break
        pos += 1
      mask_file = f'{self.base_dir}/{type}/{name:06d}_{pos:06d}.png'
      if not os.path.exists(mask_file):
        logging.info(f'{mask_file} not found')
        return None
    else:
      # mask_dir = os.path.dirname(self.color_files[0]).replace('rgb',type)
      # mask_file = f'{mask_dir}/{self.id_strs[i_frame]}_{ob_id:06d}.png'
      raise RuntimeError
    mask = cv2.imread(mask_file, -1)
    if self.resize!=1:
      mask = cv2.resize(mask, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)
    return mask>0


  def get_gt_mesh(self, ob_id:int):
    mesh_file = self.get_gt_mesh_file(ob_id)
    mesh = trimesh.load(mesh_file)
    mesh.vertices *= 1e-3
    return mesh


  def get_model_diameter(self, ob_id):
    dir = os.path.dirname(self.get_gt_mesh_file(self.ob_ids[0]))
    info_file = f'{dir}/models_info.json'
    with open(info_file,'r') as ff:
      info = json.load(ff)
    return info[str(ob_id)]['diameter']/1e3



  def get_gt_poses(self, i_frame, ob_id):
    gt_poses = []
    name = int(self.id_strs[i_frame])
    for i_k, k in enumerate(self.scene_gt[str(name)]):
      if k['obj_id']==ob_id:
        cur = np.eye(4)
        cur[:3,:3] = np.array(k['cam_R_m2c']).reshape(3,3)
        cur[:3,3] = np.array(k['cam_t_m2c'])/1e3
        gt_poses.append(cur)
    return np.asarray(gt_poses).reshape(-1,4,4)


  def get_gt_pose(self, i_frame:int, ob_id, mask=None, use_my_correction=False):
    ob_in_cam = np.eye(4)
    best_iou = -np.inf
    best_gt_mask = None
    name = int(self.id_strs[i_frame])
    for i_k, k in enumerate(self.scene_gt[str(name)]):
      if k['obj_id']==ob_id:
        cur = np.eye(4)
        cur[:3,:3] = np.array(k['cam_R_m2c']).reshape(3,3)
        cur[:3,3] = np.array(k['cam_t_m2c'])/1e3
        if mask is not None:  # When multi-instance exists, use mask to determine which one
          gt_mask = cv2.imread(f'{self.base_dir}/mask_visib/{self.id_strs[i_frame]}_{i_k:06d}.png', -1).astype(bool)
          intersect = (gt_mask*mask).astype(bool)
          union = (gt_mask+mask).astype(bool)
          iou = float(intersect.sum())/union.sum()
          if iou>best_iou:
            best_iou = iou
            best_gt_mask = gt_mask
            ob_in_cam = cur
        else:
          ob_in_cam = cur
          break


    if use_my_correction:
      if 'ycb' in self.base_dir.lower() and 'train_real' in self.color_files[i_frame]:
        video_id = self.get_video_id()
        if ob_id==1:
          if video_id in [12,13,14,17,24]:
            ob_in_cam = ob_in_cam@self.symmetry_tfs[ob_id][1]
    return ob_in_cam


  def load_symmetry_tfs(self):
    dir = os.path.dirname(self.get_gt_mesh_file(self.ob_ids[0]))
    info_file = f'{dir}/models_info.json'
    with open(info_file,'r') as ff:
      info = json.load(ff)
    self.symmetry_tfs = {}
    self.symmetry_info_table = {}
    for ob_id in self.ob_ids:
      self.symmetry_info_table[ob_id] = info[str(ob_id)]
      self.symmetry_tfs[ob_id] = symmetry_tfs_from_info(info[str(ob_id)], rot_angle_discrete=5)
    self.geometry_symmetry_info_table = copy.deepcopy(self.symmetry_info_table)


  def get_video_id(self):
    return int(self.base_dir.split('/')[-1])


class LinemodOcclusionReader(BopBaseReader):
  def __init__(self,base_dir='/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD-O/lmo_test_all/test/000002', zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'lmo'
    self.K = list(self.K_table.values())[0]
    # self.ob_ids = [1,5,6,8,9,10,11,12,13,14,15]
    self.ob_ids = [1,5,6,8,9,10,11,12]
    self.ob_id_to_names = {
      1: 'ape',
      2: 'benchvise',
      3: 'bowl',
      4: 'camera',
      5: 'water_pour',
      6: 'cat',
      7: 'cup',
      8: 'driller',
      9: 'duck',
      10: 'eggbox',
      11: 'glue',
      12: 'holepuncher',
      13: 'iron',
      14: 'lamp',
      15: 'phone',
    }
    self.load_symmetry_tfs()
    _ = self.make_scene_ob_ids_dict()

  def get_gt_mesh_file(self, ob_id):
    mesh_dir = f'{self.base_dir}/models/obj_{ob_id:06d}.ply'
    print(mesh_dir)
    return mesh_dir



class LinemodReader(LinemodOcclusionReader):
  def __init__(self, base_dir='/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD/lm_test_all/test/000002', zfar=np.inf, split=None):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'lm'
    if split is not None:  # train/test
      with open(f'/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD/Linemod_preprocessed/data/{self.get_video_id():02d}/{split}.txt','r') as ff:
        lines = ff.read().splitlines()
      self.color_files = []
      for line in lines:
        id = int(line)
        self.color_files.append(f'{self.base_dir}/rgb/{id:06d}.png')
      self.make_id_strs()

    self.ob_ids = np.setdiff1d(np.arange(1,16), np.array([7,3])).tolist()  # Exclude bowl and mug
    self.load_symmetry_tfs()


  def get_gt_mesh_file(self, ob_id):
    root = self.base_dir
    while 1:
      if os.path.exists(f'{root}'):
        mesh_dir = f'{root}/../../models/obj_{ob_id:06d}.ply'
        break
      else:
        root = os.path.abspath(f'{root}/../')
    return mesh_dir


  def get_reconstructed_mesh(self, ob_id, ref_view_dir):
    mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
    return mesh


class YcbVideoReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'ycbv'
    self.K = list(self.K_table.values())[0]

    self.make_id_strs()

    self.ob_ids = np.arange(1,22).astype(int).tolist()
    YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
    names = sorted(os.listdir(f'{YCB_VIDEO_DIR}/models/'))
    self.ob_id_to_names = {}
    self.name_to_ob_id = {}
    for i,ob_id in enumerate(self.ob_ids):
      self.ob_id_to_names[ob_id] = names[i]
      self.name_to_ob_id[names[i]] = ob_id

    if 'BOP' not in self.base_dir:
      with open(f'{self.base_dir}/../../keyframe.txt','r') as ff:
        self.keyframe_lines = ff.read().splitlines()

    self.load_symmetry_tfs()
    for ob_id in self.ob_ids:
      if ob_id in [1,4,6,18]:   # Cylinder
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_continuous': [
              {'axis':[0,0,1], 'offset':[0,0,0]},
            ],
          'symmetries_discrete': euler_matrix(0, np.pi, 0).reshape(1,4,4).tolist(),
          }
      elif ob_id in [13]:
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_continuous': [
              {'axis':[0,0,1], 'offset':[0,0,0]},
            ],
          }
      elif ob_id in [2,3,9,21]:   # Rectangle box
        tfs = []
        for rz in [0, np.pi]:
          for rx in [0,np.pi]:
            for ry in [0,np.pi]:
              tfs.append(euler_matrix(rx, ry, rz))
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_discrete': np.asarray(tfs).reshape(-1,4,4).tolist(),
          }
      else:
        pass

  def get_gt_mesh_file(self, ob_id):
    if 'BOP' in self.base_dir:
      mesh_file = os.path.abspath(f'{self.base_dir}/../../models/obj_{ob_id:06d}.ply')
    else:
      mesh_file = f'{self.base_dir}/../../models/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_mesh(self, ob_id:int, get_posecnn_version=False):
    if get_posecnn_version:
      YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
      mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{self.ob_id_to_names[ob_id]}/textured_simple.obj')
      return mesh
    mesh_file = self.get_gt_mesh_file(ob_id)
    mesh = trimesh.load(mesh_file, process=False)
    mesh.vertices *= 1e-3
    tex_file = mesh_file.replace('.ply','.png')
    if os.path.exists(tex_file):
      from PIL import Image
      im = Image.open(tex_file)
      uv = mesh.visual.uv
      material = trimesh.visual.texture.SimpleMaterial(image=im)
      color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
      mesh.visual = color_visuals
    return mesh


  def get_reconstructed_mesh(self, ob_id, ref_view_dir):
    mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
    return mesh


  def get_transform_reconstructed_to_gt_model(self, ob_id):
    out = np.eye(4)
    return out


  def get_visible_cloud(self, ob_id):
    file = os.path.abspath(f'{self.base_dir}/../../models/{self.ob_id_to_names[ob_id]}/visible_cloud.ply')
    pcd = o3d.io.read_point_cloud(file)
    return pcd


  def is_keyframe(self, i):
    color_file = self.color_files[i]
    video_id = self.get_video_id()
    frame_id = int(os.path.basename(color_file).split('.')[0])
    key = f'{video_id:04d}/{frame_id:06d}'
    return (key in self.keyframe_lines)



class TlessReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'tless'
    self.K = list(self.K_table.values())[0]
    # self.ob_ids = np.arange(1,31).astype(int).tolist()
    self.ob_ids = np.arange(1,20).astype(int).tolist()
    self.load_symmetry_tfs()

  # def is_keyframe(self, i):
  #   color_file = self.color_files[i]
  #   video_id = self.get_video_id()
  #   frame_id = int(os.path.basename(color_file).split('.')[0])
  #   key = f'{video_id:04d}/{frame_id:06d}'
  #   return (key in self.keyframe_lines)


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../models_cad/obj_{ob_id:06d}.ply'
    return mesh_file

  def color_index_from_image_id(self, image_id: int) -> int:
    target_name = f"{image_id:06d}.png"  # 帧文件名，比如 "000127.png"
    for idx, filepath in enumerate(self.color_files):
        if os.path.basename(filepath) == target_name:
            return idx
    raise ValueError(f"Image ID {image_id:06d} not found in color_files")



  def get_gt_mesh(self, ob_id):
    mesh = trimesh.load(self.get_gt_mesh_file(ob_id))
    mesh.vertices *= 1e-3
    mesh = trimesh_add_pure_colored_texture(mesh, color=np.ones((3))*200)
    return mesh


class HomebrewedReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'hb'
    self.ob_ids = np.arange(1,34).astype(int).tolist()
    self.load_symmetry_tfs()
    _ = self.make_scene_ob_ids_dict()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../hb_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_pose(self, i_frame:int, ob_id, use_my_correction=False):
    #logging.info("WARN HomeBrewed doesn't have GT pose")
    return np.eye(4)



class ItoddReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'itodd'
    self.make_id_strs()

    self.ob_ids = np.arange(1,29).astype(int).tolist()
    self.load_symmetry_tfs()
    _ = self.make_scene_ob_ids_dict()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../itodd_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


class IcbinReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'icbin'
    self.ob_ids = np.arange(1,3).astype(int).tolist()
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../icbin_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


class TudlReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'tudl'
    self.ob_ids = np.arange(1,4).astype(int).tolist()
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../tudl_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


class Real275Reader(BopBaseReader):
    def __init__(self, base_dir, zfar=np.inf):
        super().__init__(base_dir, zfar=zfar)
        self.dataset_name = 'real275'
        self.ob_ids = [1, 2, 3, 4, 5, 6] # NOCS 的 6 个类别
        self.load_symmetry_tfs()

    def get_gt_mesh_file(self, ob_id):
        return f'{self.base_dir}/../../../real275_models/models/obj_{ob_id:06d}.ply'


class Real275RawReader:
    def __init__(self, video_dir, zfar=np.inf):
        self.video_dir = video_dir
        self.zfar = zfar
        self.color_files = sorted(glob.glob(f"{self.video_dir}/*_color.png"))
        # NOCS 的内参通常需要从 meta.txt 或特定的 intrinsics.json 读取
        self.K = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]]) 

    def get_color(self, i):
        return imageio.imread(self.color_files[i])[..., :3]

    def get_depth(self, i):
        depth = cv2.imread(self.color_files[i].replace('_color.png', '_depth.png'), -1)
        return depth.astype(np.float32) / 1000.0 # 假设单位是mm

class TyolReader:
    def __init__(self, video_dir, zfar=2.0):
        """
        Args:
            video_dir: 具体的场景目录，例如 /path/to/tyol/test/000001
            zfar: 深度截止距离
        """
        self.video_dir = video_dir
        self.zfar = zfar
        
        # 1. 基础路径解析
        # 假设 video_dir 是 .../tyol/test/000001，那么 bop_root 就是向上三级
        self.bop_root = os.path.dirname(os.path.dirname(os.path.dirname(video_dir)))
        self.scene_id = os.path.basename(video_dir)
        
        # 2. 加载元数据 (BOP 格式标准文件)
        with open(os.path.join(video_dir, 'scene_gt.json'), 'r') as f:
            self.scene_gt = json.load(f)
        with open(os.path.join(video_dir, 'scene_camera.json'), 'r') as f:
            self.scene_camera = json.load(f)
            
        # 3. 加载文件列表
        self.color_files = sorted(glob(f"{video_dir}/rgb/*.png") + glob(f"{video_dir}/rgb/*.jpg"))
        self.depth_files = sorted(glob(f"{video_dir}/depth/*.png"))
        self.id_strs = [os.path.splitext(os.path.basename(f))[0] for f in self.color_files]
        
        # 4. 获取场景内参 (假设整个序列内参一致，取第一帧)
        first_cam = self.scene_camera['0']
        self.K = np.array(first_cam['cam_K']).reshape(3, 3)
        self.depth_scale = first_cam.get('depth_scale', 1.0)
        
        # 获取图像尺寸
        sample_img = self.get_color(0)
        self.H, self.W = sample_img.shape[:2]

        # 5. 获取场景中出现的所有物体 ID
        self.ob_ids = []
        for frame_id in self.scene_gt:
            for ann in self.scene_gt[frame_id]:
                if ann['obj_id'] not in self.ob_ids:
                    self.ob_ids.append(ann['obj_id'])
        
        # 6. 加载对称性信息 (从 models_info.json)
        self.symmetry_tfs = self._load_symmetries()

    def _load_symmetries(self):
        """加载 BOP 格式的对称性矩阵"""
        sym_dict = {}
        models_info_path = os.path.join(self.bop_root, 'models', 'models_info.json')
        if os.path.exists(models_info_path):
            with open(models_info_path, 'r') as f:
                models_info = json.load(f)
            for obj_id_str, info in models_info.items():
                # 这里可以根据 BOP 标准解析 symmetries_discrete
                # FoundationPose 通常需要 4x4 的 symmetry_tfs 列表
                sym_dict[int(obj_id_str)] = None # 简化处理，实际可按需解析
        return sym_dict

    def __len__(self):
        return len(self.color_files)

    def get_video_id(self):
        return self.scene_id

    def get_color(self, i):
        img = cv2.imread(self.color_files[i])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_depth(self, i):
        # BOP 深度通常是 uint16，单位 mm
        depth = cv2.imread(self.depth_files[i], -1).astype(np.float32)
        depth = depth * self.depth_scale / 1000.0  # 转为米 (m)
        depth[depth > self.zfar] = 0
        return depth

    def get_gt_pose(self, i, ob_id):
        """获取第 i 帧中指定 ob_id 的位姿，返回 4x4 矩阵 (单位: 米)"""
        frame_id_str = str(int(self.id_strs[i])) # BOP 通常用 "0", "1" 这种字符串
        if frame_id_str not in self.scene_gt:
            return None
        
        for ann in self.scene_gt[frame_id_str]:
            if ann['obj_id'] == ob_id:
                R = np.array(ann['cam_R_m2c']).reshape(3, 3)
                t = np.array(ann['cam_t_m2c']).reshape(3, 1) / 1000.0 # mm -> m
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3:4] = t
                return pose
        return None

    def get_mask(self, i, ob_id):
        """加载掩码。BOP 格式中通常有 mask 和 mask_visib"""
        # 注意：BOP 的 mask 文件命名通常是 {frame_id}_{instance_id}.png
        # 这里假设一个物体在场景中只有一个实例
        mask_pattern = os.path.join(self.video_dir, 'mask_visib', f"{int(self.id_strs[i]):06d}_*.png")
        mask_files = glob(mask_pattern)
        
        # 实际逻辑需根据 scene_gt 中的顺序匹配 instance_id
        # 简单处理：寻找对应 obj_id 的那个 mask
        frame_id_str = str(int(self.id_strs[i]))
        for inst_idx, ann in enumerate(self.scene_gt[frame_id_str]):
            if ann['obj_id'] == ob_id:
                mask_file = os.path.join(self.video_dir, 'mask_visib', f"{int(frame_id_str):06d}_{inst_idx:06d}.png")
                if os.path.exists(mask_file):
                    return cv2.imread(mask_file, -1)
        return None

    def get_gt_mesh(self, ob_id):
        """加载物体的 3D 模型"""
        mesh_path = os.path.join(self.bop_root, 'tyol/models', f"obj_{ob_id:06d}.ply")
        mesh = trimesh.load(mesh_path)
        # 注意：BOP 模型单位是 mm，FoundationPose 内部需要统一为 m
        mesh.vertices /= 1000.0
        return mesh
    
    def make_scene_ob_ids_dict(self):
        """适配你 run_pose_estimation 中的调用"""
        scene_dict = {}
        for i, id_str in enumerate(self.id_strs):
            scene_dict[id_str] = self.ob_ids
        return scene_dict

class Real275Reader:
    def __init__(self, video_dir, gt_pkl_path=None):
        """
        video_dir: 具体的场景路径, 如 '.../real_test/scene_1'
        gt_pkl_path: 对应的 GT pkl 文件路径 (通常在 ../gt/real_test 目录下)
        """
        self.video_dir = Path(video_dir)
        self.scene_id = self.video_dir.name # e.g., 'scene_1'
        
        # 1. 扫描所有帧 ID
        self.color_files = sorted(list(self.video_dir.glob("*_color.png")))
        self.id_strs = [f.name.split('_')[0] for f in self.color_files]
        
        # 2. 加载该场景的 Ground Truth (NOCS 官方提供的 pkl)
        self.gt_data = None
        if gt_pkl_path and os.path.exists(gt_pkl_path):
            with open(gt_pkl_path, 'rb') as f:
                self.gt_data = pickle.load(f)
        
        # 3. 标准内参 (NOCS Real275 固定值)
        self.K = np.array([[591.0125, 0, 322.525], 
                           [0, 590.16775, 244.11084], 
                           [0, 0, 1]], dtype=np.float32)
        
        # 4. 类别映射 (用于 FoundationPose 匹配模型)
        self.cat_id_to_name = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop', 6: 'mug'}
        
        # 预加载当前场景中出现的所有物体 ID
        self.ob_ids = self._get_all_ob_ids()

    def _get_all_ob_ids(self):
        """解析 meta.txt 获取当前 scene 中存在的所有实例 ID"""
        all_ids = set()
        for f_id in self.id_strs:
            meta_p = self.video_dir / f"{f_id}_meta.txt"
            if meta_p.exists():
                with open(meta_p, 'r') as f:
                    for line in f:
                        all_ids.add(int(line.strip().split(' ')[0]))
        return sorted(list(all_ids))

    def __len__(self):
        return len(self.id_strs)

    def get_color(self, i):
        return cv2.imread(str(self.color_files[i]))[..., ::-1]

    def get_depth(self, i):
        # NOCS 深度图是 16-bit PNG (单位 mm)
        depth = cv2.imread(str(self.video_dir / f"{self.id_strs[i]}_depth.png"), cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 1000.0 # 转换为米

    def get_mask(self, i, ob_id):
        """从 _mask.png 中提取特定物体的掩码"""
        mask_path = self.video_dir / f"{self.id_strs[i]}_mask.png"
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_ANYDEPTH)
        return (mask_img == ob_id).astype(np.uint8)

    def get_gt_pose(self, i, ob_id):
        """
        从 pkl 中提取 4x4 位姿。
        注意：Real275 是类别级，GT 包含 Scale。FoundationPose 注册时通常需要 scale-normalized pose。
        """
        if self.gt_data is None:
            return np.eye(4)
        
        # 根据 pkl 结构提取位姿 (这里需根据你手头 pkl 的具体 key 调整)
        # NOCS pkl 通常按帧索引，每帧包含多个物体的 gt_RTs
        try:
            # 找到对应帧的索引
            frame_idx = int(self.id_strs[i]) 
            # 寻找该帧中对应 ob_id 的索引
            inst_idx = np.where(self.gt_data['gt_handle_visibility'][frame_idx] == ob_id)[0][0]
            pose = self.gt_data['gt_RTs'][frame_idx][inst_idx] # 4x4 matrix
            return pose
        except:
            return np.eye(4)