#pragma once

#include "RenderAPI.h"
#include "misc/STL.h"
#include "misc/Singleton.h"
#include "misc/Traits.h"

#include <mutex>

// class Transform;
namespace Moer::ECS {
// The Entity is a global unique persistent identifier within the entity-component system
//	It can be stored and used for the duration of the application
//	The entity can be a different value on a different run of the application, if it was serialized
//	It must be only serialized with the SerializeEntity() function if persistence is needed across different program runs,
//		this will ensure that entities still match with their components correctly after serialization
using Entity                                  = uint64;
inline static constexpr Entity invalid_entity = 0;
// Runtime can create a new entity with this
inline Entity CreateEntity() {
    static std::atomic<Entity> next{invalid_entity + 1};
    return next.fetch_add(1);
}

} // namespace Moer::ECS
