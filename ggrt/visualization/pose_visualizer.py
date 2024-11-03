import torch

from ggrt.pose_util import to_hom


def get_camera_mesh(pose, depth=1):
    vertices = torch.tensor([[-0.5, -0.5, 1],
                             [ 0.5, -0.5, 1],
                             [ 0.5,  0.5, 1],
                             [-0.5,  0.5, 1],
                             [   0,    0, 0]]) * depth
    
    faces = torch.tensor([[0, 1, 2],
                          [0, 2, 3],
                          [0, 1, 4],
                          [1, 2, 4],
                          [2, 3, 4],
                          [3, 0, 4]])
    
    # vertices = camera.cam2world(vertices[None], pose)
    vertices = to_hom(vertices[None]) @ pose.transpose(-1, -2)

    wire_frame = vertices[:, [0,1,2,3,0,4,1,2,4,3]]
    
    return vertices, faces, wire_frame


def merge_wire_frames(wire_frame):
    wire_frame_merged = [[], [], []]
    for w in wire_frame:
        wire_frame_merged[0] += [float(n) for n in w[:, 0]] + [None]
        wire_frame_merged[1] += [float(n) for n in w[:, 1]] + [None]
        wire_frame_merged[2] += [float(n) for n in w[:, 2]] + [None]
    
    return wire_frame_merged


def merge_meshes(vertices, faces):
    mesh_N, vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)], dim=0)
    vertices_merged = vertices.view(-1, vertices.shape[-1])
    
    return vertices_merged,faces_merged


def merge_centers(centers):
    center_merged = [[], [], []]
    
    for c1, c2 in zip(*centers):
        center_merged[0] += [float(c1[0]), float(c2[0]), None]
        center_merged[1] += [float(c1[1]), float(c2[1]), None]
        center_merged[2] += [float(c1[2]), float(c2[2]), None]
    
    return center_merged


@torch.no_grad()
def visualize_cameras(vis, step, poses=[], cam_depth=0.5, colors=["blue", "magenta"], plot_dist=True):
    win_name = "gt_pred"
    data = []
    
    # set up plots
    centers = []
    for pose, color in zip(poses, colors):
        pose = pose.detach().cpu()
        vertices, faces, wire_frame = get_camera_mesh(pose, depth=cam_depth)
        center = vertices[:, -1]
        centers.append(center)
        
        # camera centers
        data.append(dict(
            type="scatter3d",
            x=[float(n) for n in center[:, 0]],
            y=[float(n) for n in center[:, 1]],
            z=[float(n) for n in center[:, 2]],
            mode="markers",
            marker=dict(color=color, size=3),
        ))
        
        # colored camera mesh
        vertices_merged, faces_merged = merge_meshes(vertices, faces)
        
        data.append(dict(
            type="mesh3d",
            x=[float(n) for n in vertices_merged[:, 0]],
            y=[float(n) for n in vertices_merged[:, 1]],
            z=[float(n) for n in vertices_merged[:, 2]],
            i=[int(n) for n in faces_merged[:, 0]],
            j=[int(n) for n in faces_merged[:, 1]],
            k=[int(n) for n in faces_merged[:, 2]],
            flatshading=True,
            color=color,
            opacity=0.05,
        ))
        
        # camera wire_frame
        wire_frame_merged = merge_wire_frames(wire_frame)
        data.append(dict(
            type="scatter3d",
            x=wire_frame_merged[0],
            y=wire_frame_merged[1],
            z=wire_frame_merged[2],
            mode="lines",
            line=dict(color=color,),
            opacity=0.3,
        ))
    
    if plot_dist:
        # distance between two poses (camera centers)
        center_merged = merge_centers(centers[:2])
        data.append(dict(
            type="scatter3d",
            x=center_merged[0],
            y=center_merged[1],
            z=center_merged[2],
            mode="lines",
            line=dict(color="red",width=4,),
        ))
        
        if len(centers)==4:
            center_merged = merge_centers(centers[2:4])
            data.append(dict(
                type="scatter3d",
                x=center_merged[0],
                y=center_merged[1],
                z=center_merged[2],
                mode="lines",
                line=dict(color="red",width=4,),
            ))
    
    # send data to visdom
    vis._send(dict(
        data=data,
        win="poses",
        eid=win_name,
        layout=dict(
            title="({})".format(step),
            autosize=True,
            margin=dict(l=30,r=30,b=30,t=30,),
            showlegend=False,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        ),
        opts=dict(title="{} poses ({})".format(win_name, step),),
    ))
