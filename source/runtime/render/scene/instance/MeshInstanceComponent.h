#pragma once
#include "scene/Entity.h"

namespace Moer::ECS {
struct MeshInstanceComponent {
    ECS::Entity mesh_entity = ECS::invalid_entity;

    // void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);
};

} // namespace Moer::ECS